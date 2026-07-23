from .positional_encoding import PositionalEncoding
from .embedding import Embedding
from .freq_embedding import FreqDimEmbedding
from .temporal_conv import TemporalConvFrontend
import torch
import torch.nn as nn
from utils import get_start_token

# Per-bin intermediate representation size used by FreqDimEmbedding.
# Fixed at 8 — large enough to capture the relationship between the channels
# at each frequency bin without making the temporal_proj layer too large.
_FREQ_EMBED_DIM = 8


class WaveHeightBaselineNN(nn.Module):
    def __init__(self,
                freqs,
                num_freqs,
                target='hs',
                num_channels=7,
                num_aux_channels=0,
                nhead=2,
                num_encoder_layers=2,
                num_decoder_layers=2,
                embed_dim=16,
                batch_first=True,
                max_len=500,
                dropout=0.1):

        super().__init__()

        self.num_freqs = num_freqs
        self.num_channels = num_channels
        self.num_aux_channels = num_aux_channels
        self.embed_dim = embed_dim
        self.target = target

        # ── Encoder embedding ────────────────────────────────────────────────
        # FreqDimEmbedding respects the (num_freqs, num_channels) structure:
        # a shared linear maps the per-bin measurement (num_channels wide)
        # into a per-bin representation, then all bins are aggregated into a
        # single embed_dim temporal token.  This gives the model a structural
        # prior that channels at the same frequency are related, producing
        # richer tokens for self-attention across longer sequences.
        self.encoder_embedding = FreqDimEmbedding(
            num_freqs=num_freqs,
            num_channels=num_channels,
            freq_embed_dim=_FREQ_EMBED_DIM,
            embed_dim=embed_dim,
            freqs=freqs,
            dropout=dropout,
        )

        # Decoder input is either scalar Hs (1-dim, flat Linear — no frequency
        # axis to structure) or a single-channel spectrum (num_freqs-dim, for
        # 'density'/'shape' targets). The latter gets the same per-bin
        # structural prior as the encoder (FreqDimEmbedding with
        # num_channels=1) instead of a flat Linear(num_freqs, embed_dim), so
        # the decoder side is treated consistently with the encoder side —
        # see decode() below for the (batch, seq, num_freqs) →
        # (batch, seq, num_freqs, 1) reshape this requires.
        self.decoder_embedding = (
            Embedding(1, embed_dim) if self.target == 'hs'
            else FreqDimEmbedding(
                num_freqs=num_freqs,
                num_channels=1,
                freq_embed_dim=_FREQ_EMBED_DIM,
                embed_dim=embed_dim,
                freqs=freqs,
                dropout=dropout,
            )
        )

        # ── Auxiliary side-input (e.g. wind) ─────────────────────────────────
        # Scalar-per-timestep channels are NOT frequency-resolved, so they
        # never go through encoder_embedding's per-freq-bin structure. Instead
        # they're embedded separately and added additively into the encoder
        # token (same convention positional encoding already uses), before PE.
        # Encoder-only: aux never reaches the decoder or get_start_token,
        # since it is never a forecast target.
        self.aux_embedding = (
            nn.Linear(num_aux_channels, embed_dim) if num_aux_channels > 0 else None
        )

        # ── Positional encoding (shared between encoder and decoder) ─────────
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_len,
                                              dropout=dropout)

        # ── Temporal conv front-end (encoder only) ───────────────────────────
        # Three dilated Conv1d layers (dilation 1, 2, 4) with pre-norm residuals
        # extract local temporal patterns at 3h/5h/9h scales before the global
        # self-attention layers.  Applied only to the encoder; the decoder is
        # short (≤ lead_time steps) and already uses a causal mask.
        self.temporal_conv = TemporalConvFrontend(embed_dim, dropout=dropout)

        # ── Transformer ──────────────────────────────────────────────────────
        # norm_first=True (pre-norm) matches temporal_conv's pre-norm residual
        # convention above and is generally more stable to train at the deeper
        # end of the search space (up to 4 encoder/decoder layers) than
        # nn.Transformer's post-norm default.
        self.transformer = nn.Transformer(
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_model=embed_dim,
            batch_first=batch_first,
            norm_first=True,
        )

        # Output layer — predict either 1 value (hs) or num_freqs (density)
        output_dim = 1 if self.target == 'hs' else num_freqs
        if self.target == 'hs':
            self.predictor = nn.Linear(embed_dim, output_dim)
        else:
            # density/shape predict a physical spectral energy density,
            # which is never negative (Hs = 4*sqrt(m0) requires m0 >= 0). A
            # plain Linear has no such constraint, so the model is only ever
            # discouraged from negative energy by the loss, never
            # prevented — Softplus makes it architecturally impossible.
            # Smooth and non-zero everywhere (unlike ReLU), so bins that
            # should carry near-zero energy still get a live gradient
            # instead of dying at exactly zero.
            self.predictor = nn.Sequential(
                nn.Linear(embed_dim, output_dim),
                nn.Softplus(),
            )


    def encode(self, src, aux=None):
        # src: (batch_size, src_seq_len, num_freqs, num_channels)
        # aux: (batch_size, src_seq_len, num_aux_channels) | None — e.g. wind
        # Structured frequency embedding → PE → temporal conv → self-attention.
        # Depends only on src/aux, so callers doing multi-step autoregressive
        # decoding should call this once and reuse the returned memory —
        # see infer() below and training_loop.py's scheduled-sampling branch.
        src = self.encoder_embedding(src)   # (batch, src_seq_len, embed_dim)
        if self.aux_embedding is not None:
            src = src + self.aux_embedding(aux)
        src = self.pos_encoder(src)
        src = self.temporal_conv(src)
        return self.transformer.encoder(src)

    def decode(self, tgt, memory):
        # tgt: (batch_size, tgt_seq_len, 1 or num_freqs)
        # memory: encoder output from encode(), (batch, src_seq_len, embed_dim)
        if self.target != 'hs':
            # FreqDimEmbedding expects an explicit (num_freqs, num_channels)
            # layout; the decoder's single-channel spectrum arrives flat, so
            # add back the size-1 channel axis.
            tgt = tgt.unsqueeze(-1)   # (batch, tgt_seq_len, num_freqs, 1)
        tgt = self.decoder_embedding(tgt)   # (batch, tgt_seq_len, embed_dim)
        tgt = self.pos_encoder(tgt)

        # Causal mask for decoder
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1)
        ).to(tgt.device)

        output = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
        return self.predictor(output)       # (batch, tgt_seq_len, output_dim)

    def forward(self, src, tgt, aux=None):
        memory = self.encode(src, aux=aux)
        return self.decode(tgt, memory)


    @torch.no_grad()
    def infer(self, src, freqs, lead_time, freq_means=None, aux=None):
        """Autoregressive inference for multi-step forecasting.

        Args:
            src        : torch.Tensor [batch_size, src_seq_len, num_freqs, num_channels]
            freqs      : torch.Tensor [num_freqs] — actual frequency grid
            lead_time  : int — number of future steps to forecast
            freq_means : torch.Tensor | None [num_freqs] — per-frequency training
                         mean μ(f) used to denormalise the spectrum before computing
                         the Hs start token (required for physically correct Hs when
                         target == 'hs'; unused for density target)
            aux        : torch.Tensor | None [batch_size, src_seq_len, num_aux_channels]
                         — auxiliary encoder side-input (e.g. wind_u/wind_v)

        Returns:
            torch.Tensor: Forecasted sequence [batch_size, lead_time, output_dim]
        """
        batch_size = src.size(0)
        output_dim = 1 if self.target == 'hs' else self.num_freqs

        # Encode src once — it never changes across decode steps, so caching
        # memory here avoids re-running the encoder (embedding, PE, temporal
        # conv, self-attention) lead_time times.
        memory = self.encode(src, aux=aux)

        output = torch.zeros((batch_size, lead_time + 1, output_dim),
                             device=src.device)

        start_token = get_start_token(src, self.target, freqs, src.device,
                                      freq_means=freq_means)
        output[:, 0] = start_token

        # freqs may be passed in on a different device (e.g. loaded on CPU
        # while the model runs on mps/cuda) — move once, outside the loop.
        freqs_dev = freqs.to(src.device)

        for i in range(lead_time):
            # Growing slice, not the full padded tensor: the causal mask
            # already blocks position i from seeing positions > i, so the
            # zero-padded tail beyond i+1 never influenced preds[:, i] — this
            # is just less work to get the same result.
            preds = self.decode(output[:, :i + 1], memory)
            step_pred = preds[:, i]

            if self.target == 'shape':
                # The decoder isn't architecturally constrained to output a
                # non-negative, unit-area spectrum, so enforce both here.
                # This only affects inference — teacher-forced training
                # still optimises against the raw decoder output
                # (nn/training_loop.py).
                #
                # Clamp to non-negative BEFORE integrating: a raw linear
                # output can have negative bins, and dividing by the *signed*
                # integral (as a previous version of this code did) lets
                # negative bins eat into the total mass, inflating the
                # rescale factor applied to every bin — including the
                # correctly-signed positive ones — so the whole spectrum's
                # shape gets distorted, not just the negative bins. Clamping
                # first means the mass reflects only real (positive)
                # content, and the negative bins are simply dropped rather
                # than allowed to cancel against positive ones.
                step_pred = step_pred.clamp(min=0.0)
                step_mass = torch.trapezoid(step_pred, freqs_dev, dim=-1)

                # Degenerate case: every bin was negative pre-clamp, so
                # step_pred is now all zeros and there's no clamped mass to
                # rescale by. Falling back to the discarded negative values
                # (as a previous version did, dividing by their own negative
                # mass) would just reintroduce the sign-cancellation this
                # clamp exists to remove. Fall back to a flat distribution
                # instead — trapz of a constant c over [freqs[0], freqs[-1]]
                # is exactly c * (freqs[-1] - freqs[0]) regardless of the
                # grid's (non-uniform) spacing, so this exactly integrates to
                # 1 and makes no claim about spectral shape.
                tiny = step_mass < 1e-8
                if tiny.any():
                    flat_value = 1.0 / (freqs_dev[-1] - freqs_dev[0])
                    flat = torch.full_like(step_pred, flat_value)
                    step_pred = torch.where(tiny.unsqueeze(-1), flat, step_pred)
                    step_mass = torch.where(tiny, torch.ones_like(step_mass), step_mass)

                step_pred = step_pred / step_mass.unsqueeze(-1)

            output[:, i + 1] = step_pred

        return output[:, 1:]
