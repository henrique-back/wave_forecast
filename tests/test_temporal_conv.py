"""
Tests for TemporalConvFrontend's padding_mode and its reuse inside
FreqDimEmbedding along the (bounded, non-cyclic) frequency axis.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nn.temporal_conv import TemporalConvFrontend
from nn.freq_embedding import FreqDimEmbedding


class TestPaddingModeDefaults:
    """Structural guard: the time-axis default must stay 'zeros', and
    FreqDimEmbedding's frequency-axis reuse must opt into 'replicate'."""

    def test_temporal_conv_frontend_defaults_to_zero_padding(self):
        frontend = TemporalConvFrontend(embed_dim=8)
        for layer in frontend.layers:
            assert layer['conv'].padding_mode == 'zeros'

    def test_freq_dim_embedding_uses_replicate_padding(self):
        freqs = torch.tensor([0.05, 0.10, 0.15, 0.20, 0.30])
        embedding = FreqDimEmbedding(
            num_freqs=len(freqs), num_channels=4,
            freq_embed_dim=8, embed_dim=16, freqs=freqs,
        )
        for layer in embedding.freq_conv.layers:
            assert layer['conv'].padding_mode == 'replicate'


class TestBoundaryArtifact:
    """Regression test for the shape_v8 bug: reusing TemporalConvFrontend's
    zero-padding along the frequency axis fabricates fake zero-energy bins
    just outside the real grid, distorting the bins nearest each edge
    regardless of the actual input. Replicate padding must not do this.

    This property is weight-independent: for a linear (bias-free) Conv1d, a
    constant input padded with its own edge value is still constant
    everywhere the kernel can reach, for any kernel weights. Zero padding
    breaks that at the boundary positions.
    """

    def _constant_conv1d(self, padding_mode, dilation, seed=0):
        torch.manual_seed(seed)
        conv = torch.nn.Conv1d(4, 4, kernel_size=3, dilation=dilation,
                                padding=dilation, padding_mode=padding_mode,
                                bias=False)
        conv.eval()
        x = torch.full((1, 4, 12), 3.0)  # constant across every position
        with torch.no_grad():
            return conv(x)

    def test_replicate_padding_preserves_constant_input(self):
        for dilation in (1, 2, 4):
            out = self._constant_conv1d('replicate', dilation)
            first, rest = out[:, :, 0], out[:, :, 1:]
            assert torch.allclose(rest, first.unsqueeze(-1).expand_as(rest), atol=1e-5), (
                f"replicate padding (dilation={dilation}) should keep a "
                f"constant input constant at every position, got {out}"
            )

    def test_zero_padding_distorts_constant_input_at_boundary(self):
        for dilation in (1, 2, 4):
            out = self._constant_conv1d('zeros', dilation)
            interior = out[:, :, dilation:-dilation]
            boundary = out[:, :, 0]
            assert not torch.allclose(
                boundary, interior[:, :, 0], atol=1e-4
            ), (
                f"expected zero padding (dilation={dilation}) to distort the "
                f"boundary position relative to the interior for a constant "
                f"input — this is the artifact replicate padding fixes"
            )

    def test_freq_dim_embedding_boundary_bins_stable_for_flat_spectrum(self):
        """End-to-end sanity check: a flat (constant-density) spectrum fed
        through the full FreqDimEmbedding pipeline should not produce
        outsized activity concentrated at the edge bins post-conv."""
        torch.manual_seed(0)
        num_freqs = 20
        freqs = torch.linspace(0.02, 0.485, num_freqs)
        embedding = FreqDimEmbedding(
            num_freqs=num_freqs, num_channels=1,
            freq_embed_dim=8, embed_dim=16, freqs=freqs,
        )
        embedding.eval()

        x = torch.full((1, 1, num_freqs, 1), 2.0)
        b, t, f, c = x.shape
        pre_conv = embedding.act(embedding.freq_proj(x.reshape(b * t, f, c)))
        pre_conv = pre_conv + embedding.freq_pos_sinusoidal + embedding.freq_pos_residual
        with torch.no_grad():
            post_conv = embedding.freq_conv(pre_conv)

        # freq_pos_sinusoidal/residual already make bins position-dependent
        # before the conv, so this isn't a constant-input case end-to-end —
        # instead assert the conv step itself doesn't blow up the edge bins
        # relative to the input scale it received.
        pre_scale = pre_conv.abs().max()
        edge_scale = torch.cat([post_conv[:, 0], post_conv[:, -1]]).abs().max()
        assert edge_scale < 10 * pre_scale, (
            f"edge bins grew implausibly large relative to input scale "
            f"({edge_scale} vs input max {pre_scale}) — possible boundary artifact"
        )
