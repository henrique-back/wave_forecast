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
    """infer()'s 'shape' branch must rescale every decoded step so its own
    trapezoidal integral over freqs is exactly 1"""

    def test_infer_shape_output_integrates_to_one(self):
        torch.manual_seed(0)
        freqs = torch.tensor(FREQS)
        num_freqs = len(FREQS)
        batch, seq_len, lead_time = 2, 5, 3

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

        # Patch decode() to return a fixed, guaranteed-positive tensor. An
        # untrained network's raw output isn't guaranteed positive, and a
        # negative-integral step would trip infer()'s near-zero-mass clamp
        # (matching utils/compute_hs.py::compute_shape's own clip
        # convention) — that's a separate, expected edge case, not what
        # this test is isolating. This test only checks infer()'s
        # renormalization arithmetic given a well-behaved decoder output.
        def fake_decode(tgt, memory):
            g = torch.Generator().manual_seed(123)
            return torch.rand(tgt.size(0), tgt.size(1), num_freqs, generator=g) + 0.1

        model.decode = fake_decode

        src = torch.rand(batch, seq_len, num_freqs, 1) + 0.1
        output = model.infer(src, freqs, lead_time)

        assert output.shape == (batch, lead_time, num_freqs)
        mass = torch.trapezoid(output, freqs, dim=-1)  # (batch, lead_time)
        assert torch.allclose(mass, torch.ones_like(mass), atol=1e-5), (
            f"Expected every predicted step to integrate to 1, got {mass}"
        )

    def test_infer_shape_handles_negative_raw_mass(self):
        """Regression test: an untrained/undertrained decoder's raw output has
        no non-negativity constraint, so its trapezoidal integral can
        legitimately be negative. A plain `.clamp(min=eps)` would force any
        such negative mass up to +eps and blow the rescaled output up by a
        factor of ~1/eps (observed ~1e10x against real data). infer() must
        preserve sign and only intervene when the mass is genuinely near
        zero, so a normal negative-mass step still rescales to exactly 1."""
        torch.manual_seed(0)
        freqs = torch.tensor(FREQS)
        num_freqs = len(FREQS)
        batch, seq_len, lead_time = 1, 3, 2

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

        # Fixed decoder output whose trapezoidal integral is moderately
        # negative (not near zero) — the case the old `.clamp(min=eps)`
        # mishandled.
        fixed_output = -torch.linspace(0.1, 1.0, num_freqs).view(1, 1, num_freqs)

        def fake_decode(tgt, memory):
            return fixed_output.expand(tgt.size(0), tgt.size(1), num_freqs)

        model.decode = fake_decode

        src = torch.rand(batch, seq_len, num_freqs, 1) + 0.1
        raw_mass = torch.trapezoid(fixed_output[0, 0], freqs).item()
        assert raw_mass < -1e-3, "test setup: expected a clearly negative raw mass"

        output = model.infer(src, freqs, lead_time)
        mass = torch.trapezoid(output, freqs, dim=-1)
        assert torch.allclose(mass, torch.ones_like(mass), atol=1e-5), (
            f"Expected negative-mass steps to still rescale to 1, got {mass}"
        )
        # And the rescaled output should just be the original divided by its
        # own (negative) mass — i.e. sign-preserved, not blown up.
        expected = fixed_output / raw_mass
        assert torch.allclose(output[0, 0], expected[0, 0], atol=1e-4)
