import torch.nn as nn


class TemporalConvFrontend(nn.Module):
    """Dilated causal convolutional front-end for the encoder.

    Three Conv1d layers at dilations 1, 2, 4 (kernel_size=3) are stacked with
    pre-norm residual connections. Effective receptive fields:
        dilation=1: 3 timesteps
        dilation=2: 5 timesteps
        dilation=4: 9 timesteps (combined stack covers 9h of hourly observations)

    Each encoder token exiting this module already carries local context from
    the surrounding window, so the subsequent Transformer self-attention layers
    can specialise on long-range dependencies rather than spending capacity on
    detecting local trends (rising swell, wind-sea growth, etc.).

    Applied to the encoder sequence only; the decoder sequence is left unchanged
    because its length equals lead_time (≤ 48 steps) and it already uses a
    causal mask.

    Args:
        embed_dim    : dimension of the token sequence (= head_dim × nhead)
        dropout      : dropout probability applied after each conv activation
        padding_mode : passed straight to nn.Conv1d. Default 'zeros' fits the
                       time axis (nothing observed before the window is a
                       reasonable fiction). Callers reusing this module along
                       a bounded, non-cyclic axis instead — e.g.
                       FreqDimEmbedding smoothing across frequency bins —
                       should pass 'replicate' so the edge bins repeat rather
                       than fabricating zero energy just outside the grid.
    """

    _DILATIONS = (1, 2, 4)

    def __init__(self, embed_dim: int, dropout: float = 0.1,
                 padding_mode: str = 'zeros'):
        super().__init__()
        self.layers = nn.ModuleList()
        for d in self._DILATIONS:
            # padding = dilation ensures same-length output for kernel_size=3:
            #   out_len = (L + 2d - d*(3-1) - 1)/1 + 1 = L  ✓
            self.layers.append(nn.ModuleDict({
                'norm': nn.LayerNorm(embed_dim),
                'conv': nn.Conv1d(embed_dim, embed_dim,
                                  kernel_size=3, dilation=d, padding=d,
                                  padding_mode=padding_mode),
            }))
        self.act     = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, embed_dim)
        Returns:
            (batch, seq_len, embed_dim)  — same shape as input
        """
        for layer in self.layers:
            residual = x
            # Pre-norm: normalise in (batch, seq_len, embed_dim) space
            normed = layer['norm'](x)
            # Conv1d expects (batch, channels, seq_len)
            normed_t = normed.transpose(1, 2)                      # (b, embed, seq)
            out = self.act(layer['conv'](normed_t))                # (b, embed, seq)
            out = self.dropout(out).transpose(1, 2)                # (b, seq, embed)
            x = residual + out
        return x
