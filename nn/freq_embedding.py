import torch.nn as nn


class FreqDimEmbedding(nn.Module):
    """Structured encoder embedding that respects the (num_freqs, num_channels) layout.

    Instead of flattening all num_freqs × num_channels values through a single
    unconstrained linear, this module first applies a shared linear across the
    num_channels dimension at each frequency bin (same weights for every bin),
    then aggregates the resulting per-bin representations into a single temporal
    token.

    This gives the model a structural prior: the 4 channels (density, alpha1,
    alpha2, r1) at a given frequency are related to each other. The resulting
    temporal token is a richer, more physically structured representation than
    a flat projection, which helps the encoder attend over longer sequences.

    Args:
        num_freqs       : number of frequency bins (e.g. 47)
        num_channels    : number of measurement channels per bin (e.g. 4)
        freq_embed_dim  : intermediate per-bin representation size
        embed_dim       : final temporal token dimension (= head_dim × nhead)
    """

    def __init__(self, num_freqs: int, num_channels: int,
                 freq_embed_dim: int, embed_dim: int):
        super().__init__()
        # Shared across all frequency bins: maps 4-channel measurement → freq_embed_dim
        self.freq_proj = nn.Linear(num_channels, freq_embed_dim)
        self.act = nn.GELU()
        # Aggregate all freq-bin representations into one temporal token
        self.temporal_proj = nn.Linear(num_freqs * freq_embed_dim, embed_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, num_freqs, num_channels)
        Returns:
            (batch, seq_len, embed_dim)
        """
        b, t, f, c = x.shape
        # Apply shared freq_proj to each (freq_bin, channel) slice
        x = x.reshape(b * t, f, c)          # (b*t, num_freqs, num_channels)
        x = self.act(self.freq_proj(x))      # (b*t, num_freqs, freq_embed_dim)
        x = x.reshape(b * t, -1)             # (b*t, num_freqs * freq_embed_dim)
        x = self.temporal_proj(x)            # (b*t, embed_dim)
        return x.view(b, t, -1)              # (batch, seq_len, embed_dim)
