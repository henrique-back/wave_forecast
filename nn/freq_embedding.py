import math

import torch
import torch.nn as nn

from .temporal_conv import TemporalConvFrontend


def _log_freq_sinusoidal_encoding(freqs: torch.Tensor, freq_embed_dim: int) -> torch.Tensor:
    """Sinusoidal encoding of each bin's actual (log) frequency value.

    Mirrors PositionalEncoding's sin/cos scheme, but keyed on log(freqs)
    instead of an integer time index — bins are non-uniformly spaced (dense
    near 0.02 Hz, coarse near 0.485 Hz), so embedding-space distance between
    two bins should track real log-frequency distance, not array position.
    Wavelengths are scaled to the actual span of log(freqs) (not the
    arbitrary base=10000 used for long integer sequences) so the lowest-order
    component varies smoothly across the whole spectrum and the highest-order
    component still resolves adjacent-bin differences.

    Returns: (num_freqs, freq_embed_dim)
    """
    log_f = torch.log(freqs)
    span = (log_f.max() - log_f.min()).clamp(min=1e-6)
    num_pairs = freq_embed_dim // 2

    min_wavelength = span / len(freqs)
    max_wavelength = span
    i = torch.arange(num_pairs, dtype=torch.float32)
    ratio = max_wavelength / min_wavelength
    wavelengths = min_wavelength * ratio ** (i / max(num_pairs - 1, 1))
    angular_freq = 2 * math.pi / wavelengths                  # (num_pairs,)

    args = log_f.unsqueeze(1) * angular_freq.unsqueeze(0)     # (num_freqs, num_pairs)
    pe = torch.zeros(len(freqs), freq_embed_dim)
    pe[:, 0::2] = torch.sin(args)
    pe[:, 1::2] = torch.cos(args)
    return pe


class _FreqAttentionPool(nn.Module):
    """Single learned query attends over the frequency-bin axis to produce one summary token.

    Content-aware alternative to flatten+Linear: the softmax weighting adapts
    to where the spectral peak actually sits on a given timestep, rather than
    applying a fixed per-bin-position weight learned once at training time.
    Parameter count depends only on freq_embed_dim, not num_freqs.
    """

    def __init__(self, freq_embed_dim: int, embed_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(freq_embed_dim) * freq_embed_dim ** -0.5)
        self.scale = freq_embed_dim ** 0.5
        self.out_proj = nn.Linear(freq_embed_dim, embed_dim)

    def forward(self, x):
        # x: (batch, num_freqs, freq_embed_dim)
        scores = (x @ self.query) / self.scale              # (batch, num_freqs)
        weights = torch.softmax(scores, dim=-1)              # (batch, num_freqs)
        pooled = torch.einsum('bf,bfd->bd', weights, x)      # (batch, freq_embed_dim)
        return self.out_proj(pooled)                         # (batch, embed_dim)


class FreqDimEmbedding(nn.Module):
    """Structured encoder embedding that respects the (num_freqs, num_channels) layout.

    Pipeline per timestep:
      1. A shared linear maps the per-bin channel measurement into a per-bin
         representation (same weights for every bin).
      2. A frequency-identity signal is added to each bin's representation: a
         fixed sinusoidal encoding of its actual (log) frequency value, plus a
         learned per-bin residual (zero-initialised, so training starts from
         the pure sinusoidal signal and only learns corrections the physical
         grid alone doesn't capture).
      3. A dilated-conv frontend (reusing TemporalConvFrontend along the
         frequency axis instead of time) lets each bin's representation
         absorb local context from its neighbours, exploiting that a wave
         spectrum is a smooth curve in frequency.
      4. A single-query attention pool collapses all bins into one embed_dim
         token, weighting bins by their actual content instead of a fixed
         per-position weight.

    Args:
        num_freqs       : number of frequency bins (e.g. 47)
        num_channels    : number of measurement channels per bin (e.g. 4)
        freq_embed_dim  : intermediate per-bin representation size
        embed_dim       : final temporal token dimension (= head_dim × nhead)
        freqs           : torch.Tensor (num_freqs,) — actual frequency grid in
                          Hz, used once at construction time to build the
                          sinusoidal frequency-identity encoding
        dropout         : dropout probability inside the conv frontend
    """

    def __init__(self, num_freqs: int, num_channels: int,
                 freq_embed_dim: int, embed_dim: int,
                 freqs: torch.Tensor, dropout: float = 0.1):
        super().__init__()
        # Shared across all frequency bins: maps 4-channel measurement → freq_embed_dim
        self.freq_proj = nn.Linear(num_channels, freq_embed_dim)
        self.act = nn.GELU()

        self.register_buffer(
            'freq_pos_sinusoidal',
            _log_freq_sinusoidal_encoding(freqs, freq_embed_dim),
        )
        self.freq_pos_residual = nn.Parameter(torch.zeros(num_freqs, freq_embed_dim))

        self.freq_conv = TemporalConvFrontend(freq_embed_dim, dropout=dropout)
        self.attn_pool = _FreqAttentionPool(freq_embed_dim, embed_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, num_freqs, num_channels)
        Returns:
            (batch, seq_len, embed_dim)
        """
        b, t, f, c = x.shape
        # Apply shared freq_proj to each (freq_bin, channel) slice
        x = x.reshape(b * t, f, c)                  # (b*t, num_freqs, num_channels)
        x = self.act(self.freq_proj(x))             # (b*t, num_freqs, freq_embed_dim)
        x = x + self.freq_pos_sinusoidal + self.freq_pos_residual
        x = self.freq_conv(x)                        # (b*t, num_freqs, freq_embed_dim)
        x = self.attn_pool(x)                         # (b*t, embed_dim)
        return x.view(b, t, -1)                       # (batch, seq_len, embed_dim)
