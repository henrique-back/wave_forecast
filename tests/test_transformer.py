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

    def test_infer_shape_clamps_negative_bins_before_mass_rescale(self):
        """Regression test: negative bins must be clamped to zero BEFORE the
        trapezoidal mass is computed, not after. Dividing by the raw
        (signed) integral lets negative bins eat into the total mass,
        inflating the rescale factor applied to every bin — including the
        correctly-signed positive ones — so the whole spectrum gets
        distorted by sign-cancellation, not just the negative bins
        themselves. infer() must instead clamp first, so the mass reflects
        only the real positive content."""
        torch.manual_seed(0)
        freqs = torch.tensor(FREQS)
        num_freqs = len(FREQS)
        batch, seq_len, lead_time = 1, 3, 1

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

        # Mixed-sign raw output: some real positive content, plus negative
        # bins that would otherwise cancel part of that mass away.
        fixed_output = torch.tensor(
            [[[2.0, 3.0, -4.0, 1.0, -1.0]]]
        )  # matches len(FREQS) == 5

        def fake_decode(tgt, memory):
            return fixed_output.expand(tgt.size(0), tgt.size(1), num_freqs)

        model.decode = fake_decode

        src = torch.rand(batch, seq_len, num_freqs, 1) + 0.1
        output = model.infer(src, freqs, lead_time)

        assert (output >= 0).all(), f"expected no negative bins in output, got {output}"
        mass = torch.trapezoid(output, freqs, dim=-1)
        assert torch.allclose(mass, torch.ones_like(mass), atol=1e-5), (
            f"Expected output to still integrate to 1, got {mass}"
        )

        # The correct rescale clamps first, then divides by the CLAMPED
        # mass — not the naive (and wrong) raw/raw_mass.
        clamped = fixed_output.clamp(min=0.0)
        clamped_mass = torch.trapezoid(clamped[0, 0], freqs)
        expected = clamped / clamped_mass
        assert torch.allclose(output[0, 0], expected[0, 0], atol=1e-4)

        # Sanity check that this genuinely differs from the old (buggy)
        # raw/raw_mass behavior for this input — otherwise the test above
        # wouldn't actually be distinguishing the two implementations.
        raw_mass = torch.trapezoid(fixed_output[0, 0], freqs)
        naive_wrong = fixed_output[0, 0] / raw_mass
        assert not torch.allclose(output[0, 0], naive_wrong, atol=1e-3)

    def test_infer_shape_falls_back_to_flat_when_all_bins_negative(self):
        """Degenerate edge case: if every bin in a step's raw output is
        negative, clamping makes the whole vector zero and there's no
        positive mass left to rescale by. Falling back to the discarded
        negative values (dividing by their own negative mass, as a previous
        implementation did) would just reintroduce the sign-cancellation
        the clamp exists to remove — infer() must fall back to a flat
        (uniform) unit-area distribution instead, preserving the
        integrates-to-1 invariant without fabricating any shape claim."""
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

        fixed_output = -torch.linspace(0.1, 1.0, num_freqs).view(1, 1, num_freqs)

        def fake_decode(tgt, memory):
            return fixed_output.expand(tgt.size(0), tgt.size(1), num_freqs)

        model.decode = fake_decode

        src = torch.rand(batch, seq_len, num_freqs, 1) + 0.1
        output = model.infer(src, freqs, lead_time)

        assert (output >= 0).all(), f"expected no negative bins in output, got {output}"
        mass = torch.trapezoid(output, freqs, dim=-1)
        assert torch.allclose(mass, torch.ones_like(mass), atol=1e-5), (
            f"Expected the flat fallback to integrate to 1, got {mass}"
        )
        flat_value = 1.0 / (freqs[-1] - freqs[0])
        assert torch.allclose(output, flat_value.expand_as(output), atol=1e-5)


class TestPredictorNonNegativity:
    """density/shape predict a physical spectral energy density, which is
    never negative — the predictor head must be architecturally incapable
    of outputting negative energy (Softplus), not just discouraged from it
    by the loss. 'hs' has no such requirement and keeps a plain Linear."""

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

    def test_density_and_shape_predictor_end_in_softplus(self):
        for target in ("density", "shape"):
            model = self._build(target)
            assert isinstance(model.predictor, torch.nn.Sequential)
            assert isinstance(model.predictor[-1], torch.nn.Softplus)

    def test_hs_predictor_stays_plain_linear(self):
        model = self._build("hs")
        assert isinstance(model.predictor, torch.nn.Linear)

    def test_density_and_shape_decode_output_always_positive(self):
        """Must hold with zero training — an architectural property of
        Softplus, not something the model has to learn."""
        num_freqs = len(FREQS)
        batch, seq_len, tgt_len = 2, 4, 3
        for target in ("density", "shape"):
            model = self._build(target)
            model.eval()
            src = torch.randn(batch, seq_len, num_freqs, 1)
            tgt = torch.randn(batch, tgt_len, num_freqs)
            memory = model.encode(src)
            output = model.decode(tgt, memory)
            assert (output > 0).all(), (
                f"expected strictly positive output for target={target!r}, "
                f"got min={output.min().item()}"
            )

    def test_hs_decode_output_can_be_negative(self):
        """Sanity check that the 'hs' branch is genuinely unconstrained —
        otherwise the contrast above wouldn't be testing anything."""
        model = self._build("hs")
        model.eval()
        num_freqs = len(FREQS)
        batch, seq_len, tgt_len = 8, 4, 5
        src = torch.randn(batch, seq_len, num_freqs, 1)
        tgt = torch.randn(batch, tgt_len, 1)
        memory = model.encode(src)
        output = model.decode(tgt, memory)
        assert (output < 0).any(), (
            "expected at least one negative value from an unconstrained "
            "random-init Linear head over this many samples"
        )
