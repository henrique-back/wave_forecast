from .positional_encoding import PositionalEncoding
from .embedding import Embedding
from .freq_embedding import FreqDimEmbedding
from .temporal_conv import TemporalConvFrontend
import torch
import torch.nn as nn
from utils import get_start_token

# Per-bin intermediate representation size used by FreqDimEmbedding.
# Fixed at 8 — large enough to capture the relationship between the 4 channels
# at each frequency bin without making the temporal_proj layer too large.
_FREQ_EMBED_DIM = 8


class WaveHeightBaselineNN(nn.Module):
    def __init__(self,
                freqs,
                num_freqs,
                target='hs',
                num_channels=4,
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
        self.embed_dim = embed_dim
        self.target = target

        # ── Encoder embedding ────────────────────────────────────────────────
        # FreqDimEmbedding respects the (num_freqs, num_channels) structure:
        # a shared linear maps the 4-channel measurement at each frequency bin
        # into a per-bin representation, then all bins are aggregated into a
        # single embed_dim temporal token.  This gives the model a structural
        # prior that channels at the same frequency are related, producing
        # richer tokens for self-attention across longer sequences.
        self.encoder_embedding = FreqDimEmbedding(
            num_freqs=num_freqs,
            num_channels=num_channels,
            freq_embed_dim=_FREQ_EMBED_DIM,
            embed_dim=embed_dim,
        )

        # Decoder input is either scalar Hs (1-dim) or flat normalised density
        # (num_freqs-dim, single channel) — no multi-channel structure here.
        self.decoder_embedding = Embedding(
            1 if self.target == 'hs' else num_freqs, embed_dim
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
        self.transformer = nn.Transformer(
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_model=embed_dim,
            batch_first=batch_first,
        )

        # Output layer — predict either 1 value (hs) or num_freqs (density)
        output_dim = 1 if self.target == 'hs' else num_freqs
        self.predictor = nn.Linear(embed_dim, output_dim)


    def encode(self, src):
        # src: (batch_size, src_seq_len, num_freqs, num_channels)
        # Structured frequency embedding → PE → temporal conv → self-attention.
        # Depends only on src, so callers doing multi-step autoregressive
        # decoding should call this once and reuse the returned memory —
        # see infer() below and training_loop.py's scheduled-sampling branch.
        src = self.encoder_embedding(src)   # (batch, src_seq_len, embed_dim)
        src = self.pos_encoder(src)
        src = self.temporal_conv(src)
        return self.transformer.encoder(src)

    def decode(self, tgt, memory):
        # tgt: (batch_size, tgt_seq_len, 1 or num_freqs)
        # memory: encoder output from encode(), (batch, src_seq_len, embed_dim)
        tgt = self.decoder_embedding(tgt)   # (batch, tgt_seq_len, embed_dim)
        tgt = self.pos_encoder(tgt)

        # Causal mask for decoder
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1)
        ).to(tgt.device)

        output = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
        return self.predictor(output)       # (batch, tgt_seq_len, output_dim)

    def forward(self, src, tgt):
        memory = self.encode(src)
        return self.decode(tgt, memory)


    @torch.no_grad()
    def infer(self, src, freqs, lead_time, freq_means=None):
        """Autoregressive inference for multi-step forecasting.

        Args:
            src        : torch.Tensor [batch_size, src_seq_len, num_freqs, num_channels]
            freqs      : torch.Tensor [num_freqs] — actual frequency grid
            lead_time  : int — number of future steps to forecast
            freq_means : torch.Tensor | None [num_freqs] — per-frequency training
                         mean μ(f) used to denormalise the spectrum before computing
                         the Hs start token (required for physically correct Hs when
                         target == 'hs'; unused for density target)

        Returns:
            torch.Tensor: Forecasted sequence [batch_size, lead_time, output_dim]
        """
        batch_size = src.size(0)
        output_dim = 1 if self.target == 'hs' else self.num_freqs

        # Encode src once — it never changes across decode steps, so caching
        # memory here avoids re-running the encoder (embedding, PE, temporal
        # conv, self-attention) lead_time times.
        memory = self.encode(src)

        output = torch.zeros((batch_size, lead_time + 1, output_dim),
                             device=src.device)

        start_token = get_start_token(src, self.target, freqs, src.device,
                                      freq_means=freq_means)
        output[:, 0] = start_token

        for i in range(lead_time):
            # Growing slice, not the full padded tensor: the causal mask
            # already blocks position i from seeing positions > i, so the
            # zero-padded tail beyond i+1 never influenced preds[:, i] — this
            # is just less work to get the same result.
            preds = self.decode(output[:, :i + 1], memory)
            output[:, i + 1] = preds[:, i]

        return output[:, 1:]
