"""
Tests for WaveHeightBaselineNN.infer()'s target-specific behaviour.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nn.transformer import WaveHeightBaselineNN


FREQS = np.array([0.05, 0.10, 0.15, 0.20, 0.30], dtype=np.float32)


class TestShapeInferRenormalization:
    """infer()'s 'shape' branch decodes log-shape, exp()'s it back to linear
    shape, rescales every step so its trapezoidal integral over freqs is
    exactly 1, then converts back to log-space before returning/feeding the
    next decode step — so every assertion here exp()'s the returned
    sequence before checking mass."""

    def test_infer_shape_output_integrates_to_one(self):
        torch.manual_seed(0)
        freqs = torch.tensor(FREQS)
        num_freqs = len(FREQS)
        batch, seq_len, lead_time = 2, 5, 3
        freq_means = torch.ones(num_freqs)
        shape_means = torch.ones(num_freqs)

        model = WaveHeightBaselineNN(
            freqs=freqs,
            num_freqs=num_freqs,
            target="shape",
            num_channels=1,
            nhead=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            embed_dim=8,
        )
        model.eval()

        # decode() now returns log-shape, which is unconstrained — a plain
        # randn is a perfectly well-behaved decoder output post-ablation,
        # since infer() exp()'s it (always > 0) before the mass arithmetic.
        def fake_decode(tgt, memory):
            g = torch.Generator().manual_seed(123)
            return torch.randn(tgt.size(0), tgt.size(1), num_freqs, generator=g)

        model.decode = fake_decode

        src = torch.rand(batch, seq_len, num_freqs, 1) + 0.1
        output = model.infer(src, freqs, lead_time, freq_means=freq_means, shape_means=shape_means)

        assert output.shape == (batch, lead_time, num_freqs)
        shape_lin = torch.exp(output)
        mass = torch.trapezoid(shape_lin, freqs, dim=-1)  # (batch, lead_time)
        assert torch.allclose(mass, torch.ones_like(mass), atol=1e-5), (
            f"Expected every predicted step to integrate to 1, got {mass}"
        )

    def test_infer_shape_rescale_matches_manual_computation(self):
        """Regression test: infer() must exp() the raw log-shape decode()
        output, rescale by the resulting linear-space mass, then log() the
        rescaled result back before storing/returning it — not skip the
        round-trip, and not rescale in log-space directly (which would be
        mathematically wrong: log(a/b) != log(a)/b)."""
        torch.manual_seed(0)
        freqs = torch.tensor(FREQS)
        num_freqs = len(FREQS)
        batch, seq_len, lead_time = 1, 3, 1
        freq_means = torch.ones(num_freqs)
        shape_means = torch.ones(num_freqs)

        model = WaveHeightBaselineNN(
            freqs=freqs,
            num_freqs=num_freqs,
            target="shape",
            num_channels=1,
            nhead=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            embed_dim=8,
        )
        model.eval()

        # Mixed-sign raw log-shape output — perfectly valid now, since exp()
        # recovers a strictly positive linear-space value regardless of sign.
        fixed_output = torch.tensor(
            [[[2.0, 3.0, -4.0, 1.0, -1.0]]]
        )  # matches len(FREQS) == 5

        def fake_decode(tgt, memory):
            return fixed_output.expand(tgt.size(0), tgt.size(1), num_freqs)

        model.decode = fake_decode

        src = torch.rand(batch, seq_len, num_freqs, 1) + 0.1
        output = model.infer(src, freqs, lead_time, freq_means=freq_means, shape_means=shape_means)

        shape_lin = torch.exp(output)
        assert (shape_lin > 0).all(), f"expected strictly positive shape, got {shape_lin}"
        mass = torch.trapezoid(shape_lin, freqs, dim=-1)
        assert torch.allclose(mass, torch.ones_like(mass), atol=1e-5), (
            f"Expected output to integrate to 1, got {mass}"
        )

        # Manual reference: exp -> mass -> rescale (all in linear space).
        linear = torch.exp(fixed_output[0, 0])
        linear_mass = torch.trapezoid(linear, freqs)
        expected_linear = linear / linear_mass
        assert torch.allclose(shape_lin[0, 0], expected_linear, atol=1e-4)

    def test_infer_shape_falls_back_to_flat_on_underflow(self):
        """Degenerate edge case: if the raw log-shape output is very
        negative for every bin, exp() underflows to (near-)zero across the
        board and there's no positive mass left to rescale by. infer() must
        fall back to a flat (uniform) unit-area distribution instead of
        dividing by ~0, preserving the integrates-to-1 invariant without
        fabricating any shape claim."""
        torch.manual_seed(0)
        freqs = torch.tensor(FREQS)
        num_freqs = len(FREQS)
        batch, seq_len, lead_time = 1, 3, 2
        freq_means = torch.ones(num_freqs)
        shape_means = torch.ones(num_freqs)

        model = WaveHeightBaselineNN(
            freqs=freqs,
            num_freqs=num_freqs,
            target="shape",
            num_channels=1,
            nhead=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            embed_dim=8,
        )
        model.eval()

        # exp(-1000) underflows to exactly 0.0 in float32.
        fixed_output = torch.full((1, 1, num_freqs), -1000.0)

        def fake_decode(tgt, memory):
            return fixed_output.expand(tgt.size(0), tgt.size(1), num_freqs)

        model.decode = fake_decode

        src = torch.rand(batch, seq_len, num_freqs, 1) + 0.1
        output = model.infer(src, freqs, lead_time, freq_means=freq_means, shape_means=shape_means)

        shape_lin = torch.exp(output)
        assert (shape_lin >= 0).all(), f"expected no negative bins, got {shape_lin}"
        mass = torch.trapezoid(shape_lin, freqs, dim=-1)
        assert torch.allclose(mass, torch.ones_like(mass), atol=1e-5), (
            f"Expected the flat fallback to integrate to 1, got {mass}"
        )
        flat_value = 1.0 / (freqs[-1] - freqs[0])
        assert torch.allclose(shape_lin, flat_value.expand_as(shape_lin), atol=1e-5)


class TestPredictorArchitecture:
    """density/shape now predict log-spectral-energy (log E(f) or
    log E(f)/m0) rather than the physical value directly — non-negativity
    of the physical quantity is guaranteed for free by exp() at the point
    of use, so the predictor head is a plain Linear for all three targets,
    same as 'hs' always was, and decode() output is unconstrained (can be
    negative) for every target."""

    def _build(self, target):
        torch.manual_seed(0)
        freqs = torch.tensor(FREQS)
        return WaveHeightBaselineNN(
            freqs=freqs,
            num_freqs=len(FREQS),
            target=target,
            num_channels=1,
            nhead=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            embed_dim=8,
        )

    def test_all_targets_use_plain_linear_predictor(self):
        for target in ("hs", "density", "shape"):
            model = self._build(target)
            assert isinstance(model.predictor, torch.nn.Linear), (
                f"expected plain Linear predictor for target={target!r}, "
                f"got {type(model.predictor)}"
            )

    def test_decode_output_can_be_negative_for_every_target(self):
        """Sanity check that no target's decode() output is architecturally
        constrained to be positive — otherwise this contrast wouldn't be
        testing anything."""
        num_freqs = len(FREQS)
        batch, seq_len, tgt_len = 8, 4, 5
        for target in ("hs", "density", "shape"):
            model = self._build(target)
            model.eval()
            src = torch.randn(batch, seq_len, num_freqs, 1)
            tgt_width = 1 if target == "hs" else num_freqs
            tgt = torch.randn(batch, tgt_len, tgt_width)
            memory = model.encode(src)
            output = model.decode(tgt, memory)
            assert (output < 0).any(), (
                f"expected at least one negative value from an unconstrained "
                f"random-init Linear head for target={target!r} over this "
                f"many samples"
            )
