"""
Tests for utils.loss.DirectionalLoss — the weighted composite loss over
spectral density + directional wave parameters proposed in Meta 3 of the
2026-07-24 meeting doc (deliverable 3.5: "test a first version of the
weighted loss").
"""
import os
import sys

import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.loss import (DirectionalLoss, SpectralWassersteinLoss,
                         SpectralKLDivergenceLoss, _trapz_bin_widths, _FULL_CHANNELS)
from tests.test_spectral import FREQS as SPECTRAL_FREQS


def _make_sample(density, alpha1_deg, alpha2_deg, r1, r2=0.5, num_freqs=4):
    """(num_freqs, 7) tensor in DirectionalLoss's expected channel order."""
    a1 = np.radians(alpha1_deg)
    a2 = np.radians(alpha2_deg)
    return torch.tensor([
        [density, np.sin(a1), np.cos(a1), np.sin(a2), np.cos(a2), r1, r2]
        for _ in range(num_freqs)
    ], dtype=torch.float32)


class TestDirectionalLossChannelOrder:
    def test_matches_nn_channels_full(self):
        """_FULL_CHANNELS must stay in sync with nn.channels.CHANNEL_SETS
        ['full'] — hardcoded locally to avoid a circular import (utils must
        not depend on nn), so this test is the drift guard."""
        from nn.channels import CHANNEL_SETS
        assert _FULL_CHANNELS == CHANNEL_SETS['full']


class TestDirectionalLossBasics:
    def test_zero_loss_for_perfect_prediction(self):
        loss_fn = DirectionalLoss()
        true = _make_sample(1.5, 45.0, 200.0, 0.6)
        total, components = loss_fn(true, true)
        assert total.item() == pytest.approx(0.0, abs=1e-6)
        for v in components.values():
            assert v.item() == pytest.approx(0.0, abs=1e-6)

    def test_total_equals_weighted_sum_of_components(self):
        loss_fn = DirectionalLoss(lambda_E=2.0, lambda_alpha1=0.5,
                                   lambda_alpha2=0.1, lambda_r=1.5)
        true = _make_sample(1.5, 45.0, 200.0, 0.6)
        pred = _make_sample(2.0, 60.0, 210.0, 0.4)
        total, c = loss_fn(true, pred)
        expected = (2.0 * c['L_E'] + 0.5 * c['L_alpha1']
                    + 0.1 * c['L_alpha2'] + 1.5 * c['L_r'])
        assert total.item() == pytest.approx(expected.item(), rel=1e-5)

    def test_lambda_zero_removes_component_contribution(self):
        """Setting a lambda to 0 must not change total even if that
        component's prediction is wildly wrong."""
        true = _make_sample(1.5, 45.0, 200.0, 0.6)
        bad_r_pred = _make_sample(1.5, 45.0, 200.0, 99.0)  # only r1 is wrong

        loss_with_r = DirectionalLoss(lambda_E=1.0, lambda_alpha1=1.0,
                                       lambda_alpha2=1.0, lambda_r=1.0)
        loss_without_r = DirectionalLoss(lambda_E=1.0, lambda_alpha1=1.0,
                                          lambda_alpha2=1.0, lambda_r=0.0)

        total_with_r, _ = loss_with_r(bad_r_pred, true)
        total_without_r, _ = loss_without_r(bad_r_pred, true)

        assert total_with_r.item() > 1.0  # r1 error dominates
        assert total_without_r.item() == pytest.approx(0.0, abs=1e-6)


class TestDirectionalLossAngularWraparound:
    """The core value proposition of sin/cos over raw-degree MSE: physically
    adjacent angles across the 0/360 boundary must produce a small loss."""

    def test_wraparound_gives_small_loss_not_large(self):
        loss_fn = DirectionalLoss()
        true = _make_sample(1.0, 1.0, 180.0, 0.5)      # alpha1 = 1 deg
        pred = _make_sample(1.0, 359.0, 180.0, 0.5)     # alpha1 = 359 deg

        # Physically 2 degrees apart, not 358.
        _, components = loss_fn(pred, true)
        # A naive raw-degree MSE would give (359-1)^2 = 128164; sin/cos
        # encoding must stay tiny for a 2-degree true separation.
        assert components['L_alpha1'].item() < 0.01

    def test_wraparound_loss_matches_small_true_separation(self):
        """1 deg vs 359 deg (sin/cos loss) should be close to 1 deg vs 3 deg
        (a small, non-wrapping separation of the same 2-degree magnitude)."""
        loss_fn = DirectionalLoss()

        true_wrap = _make_sample(1.0, 1.0, 180.0, 0.5)
        pred_wrap = _make_sample(1.0, 359.0, 180.0, 0.5)
        _, c_wrap = loss_fn(pred_wrap, true_wrap)

        true_nowrap = _make_sample(1.0, 1.0, 180.0, 0.5)
        pred_nowrap = _make_sample(1.0, 3.0, 180.0, 0.5)
        _, c_nowrap = loss_fn(pred_nowrap, true_nowrap)

        assert c_wrap['L_alpha1'].item() == pytest.approx(
            c_nowrap['L_alpha1'].item(), rel=0.05
        )


class TestDirectionalLossFreqWeighting:
    def test_freq_weights_change_result_when_nonuniform(self):
        loss_fn = DirectionalLoss()
        num_freqs = 4
        true = _make_sample(1.5, 45.0, 200.0, 0.6, num_freqs=num_freqs)
        pred = true.clone()
        pred[0, 0] += 1.0  # perturb density only at the first bin

        flat_total, _ = loss_fn(pred, true)

        # Weight the first bin heavily relative to the rest.
        weights = torch.tensor([0.7, 0.1, 0.1, 0.1])
        weighted_total, _ = loss_fn(pred, true, freq_weights=weights)

        assert weighted_total.item() > flat_total.item()

    def test_batched_multistep_shape(self):
        """(batch, lead_time, num_freqs, 7) — the shape a decoder output
        would actually have — must work, not just the bare (num_freqs, 7)
        used in the other tests here."""
        loss_fn = DirectionalLoss()
        batch, lead_time, num_freqs = 3, 5, 6
        true = torch.stack([
            _make_sample(1.5, 45.0, 200.0, 0.6, num_freqs=num_freqs)
            for _ in range(batch * lead_time)
        ]).reshape(batch, lead_time, num_freqs, 7)
        pred = true + 0.1

        total, components = loss_fn(pred, true)
        assert total.dim() == 0
        for v in components.values():
            assert v.dim() == 0


def _spike(freqs_len, center_idx, height=10.0, floor=1e-3):
    """A log-space single-bin spike (all other bins at `floor`) — a
    convenient narrow synthetic 'peak' for testing position-shift behavior."""
    x = np.full(freqs_len, floor)
    x[center_idx] = height
    return torch.log(torch.tensor(x, dtype=torch.float32))


class TestSpectralWassersteinLoss:
    """SpectralWassersteinLoss (nn/training_loop.py's Wasserstein-distance
    auxiliary term / nn/evaluate.py's always-on 'Shape_Wasserstein' metric)
    — the properties below are what distinguish it from a pointwise loss
    like RMSELoss: forgiving of small position shifts, mass-scale invariant
    (it compares shape, not magnitude)."""

    def test_zero_for_identical_spectra(self):
        loss_fn = SpectralWassersteinLoss()
        true = _spike(len(SPECTRAL_FREQS), 20)
        freqs = torch.tensor(SPECTRAL_FREQS)
        loss = loss_fn(true, true, freqs)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_mass_scale_invariant(self):
        """Scaling a prediction by a constant factor (adding a constant in
        log-space) must not change the loss — it compares normalized shape,
        not magnitude, by design (mass error is tracked separately by
        Shape_Mass_Error)."""
        loss_fn = SpectralWassersteinLoss()
        freqs = torch.tensor(SPECTRAL_FREQS)
        true = _spike(len(SPECTRAL_FREQS), 20)
        pred = _spike(len(SPECTRAL_FREQS), 30)  # shifted, so loss isn't trivially 0

        loss_unscaled = loss_fn(pred, true, freqs).item()
        loss_scaled = loss_fn(pred + np.log(5.0), true, freqs).item()  # 5x in linear space
        assert loss_scaled == pytest.approx(loss_unscaled, rel=1e-4)

    def test_forgiving_of_small_shifts_unlike_pointwise_error(self):
        """A narrow peak shifted by a small vs. large number of bins: W1
        must scale smoothly/monotonically with shift distance, while a
        plain pointwise error on the same pairs is roughly CONSTANT (a
        narrow peak has near-total non-overlap even for a 1-bin shift, so
        pointwise error can't distinguish a near miss from a far one) —
        this is the exact property motivating W1 over RMSELoss/the
        (reverted) SpectralSlopeLoss for the multimodal-blur problem."""
        loss_fn = SpectralWassersteinLoss()
        freqs = torch.tensor(SPECTRAL_FREQS)
        n = len(SPECTRAL_FREQS)
        true = _spike(n, 20)

        def pointwise_mse(pred_log):
            pred_n = torch.exp(pred_log) / torch.exp(pred_log).sum()
            true_n = torch.exp(true) / torch.exp(true).sum()
            return ((pred_n - true_n) ** 2).mean().item()

        w1_near = loss_fn(_spike(n, 21), true, freqs).item()   # shift by 1 bin
        w1_mid = loss_fn(_spike(n, 25), true, freqs).item()    # shift by 5 bins
        w1_far = loss_fn(_spike(n, 45), true, freqs).item()    # shift by 25 bins

        assert w1_near < w1_mid < w1_far  # W1 scales with distance

        mse_near = pointwise_mse(_spike(n, 21))
        mse_far = pointwise_mse(_spike(n, 45))
        assert mse_near == pytest.approx(mse_far, rel=0.05)  # pointwise error is blind to distance

    def test_respects_nonuniform_grid(self):
        """A uniform (constant-height) true and pred spectrum on a
        deliberately uneven grid must still give zero loss (sanity: the
        cumulative integration doesn't introduce spurious error from grid
        non-uniformity alone), and a perturbed pred must give a loss that
        changes with WHERE on the grid the perturbation sits, not just its
        bin-index position — a pure index-based (non-Δf-aware) computation
        would be insensitive to this."""
        loss_fn = SpectralWassersteinLoss()
        freqs = torch.tensor([0.02, 0.03, 0.07, 0.08, 0.20, 0.485])
        flat_true = torch.log(torch.full((6,), 1.0))
        assert loss_fn(flat_true, flat_true, freqs).item() == pytest.approx(0.0, abs=1e-5)

        pred_a = torch.log(torch.tensor([1.0, 1.0, 2.0, 1.0, 1.0, 1.0]))  # bump in a narrow bin
        pred_b = torch.log(torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0, 1.0]))  # bump in a wide bin
        loss_a = loss_fn(pred_a, flat_true, freqs).item()
        loss_b = loss_fn(pred_b, flat_true, freqs).item()
        assert loss_a != pytest.approx(loss_b, rel=1e-3)

    def test_batched_multistep_shape(self):
        """(batch, lead_time, num_freqs) — the shape train_one_epoch's
        y_pred/y_batch actually have — must work, not just a bare 1-D
        spectrum."""
        loss_fn = SpectralWassersteinLoss()
        freqs = torch.tensor(SPECTRAL_FREQS)
        batch, lead_time = 3, 4
        true_1d = _spike(len(SPECTRAL_FREQS), 20)
        true = true_1d.expand(batch, lead_time, -1)
        pred = true.clone()

        loss = loss_fn(pred, true, freqs)
        assert loss.dim() == 0
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_reduction_none_matches_reduction_mean(self):
        """reduction='none' (used by nn/evaluate.py's 'density' block, which
        must mask out near-zero-mass samples before averaging) must return
        the same per-sample values that reduction='mean' (the training-loss
        default) averages over — not a different computation path."""
        loss_fn = SpectralWassersteinLoss()
        freqs = torch.tensor(SPECTRAL_FREQS)
        n = len(SPECTRAL_FREQS)
        true = torch.stack([_spike(n, 15), _spike(n, 20), _spike(n, 30)])
        pred = torch.stack([_spike(n, 16), _spike(n, 20), _spike(n, 35)])

        per_sample = loss_fn(pred, true, freqs, reduction='none')
        scalar = loss_fn(pred, true, freqs, reduction='mean')

        assert per_sample.shape == (3,)
        assert per_sample.mean().item() == pytest.approx(scalar.item(), rel=1e-5)
        # Identical pair (index 1) must be exactly zero; the others must not.
        assert per_sample[1].item() == pytest.approx(0.0, abs=1e-5)
        assert per_sample[0].item() > 0.0
        assert per_sample[2].item() > 0.0

    def test_reduction_invalid_raises(self):
        loss_fn = SpectralWassersteinLoss()
        freqs = torch.tensor(SPECTRAL_FREQS)
        true = _spike(len(SPECTRAL_FREQS), 20)
        with pytest.raises(ValueError, match="reduction"):
            loss_fn(true, true, freqs, reduction='sum')


class TestSpectralKLDivergenceLoss:
    """SpectralKLDivergenceLoss (utils/loss.py) — KL divergence between
    predicted/true spectra as Δf-weighted probability distributions over
    frequency. Gradient-equivalent to a plain cross-entropy term (they
    differ only by H(P_true), which doesn't depend on the prediction), but
    reports exactly 0 at a perfect match, like SpectralWassersteinLoss.
    Distinguishing properties vs. SpectralWassersteinLoss: no spatial/shift
    tolerance (a coherent shift is as bad as under a pointwise loss), but
    comparatively lenient toward a peak BROADENING into its neighbourhood
    while still covering the true bin — the opposite of log-space MSE,
    which punishes broadening even harder than a full shift."""

    def test_zero_for_identical_spectra(self):
        """D_KL(P, P) = 0 exactly (Gibbs' inequality) — same convention as
        SpectralWassersteinLoss, unlike a plain cross-entropy term (which
        would bottom out at P's own nonzero entropy)."""
        loss_fn = SpectralKLDivergenceLoss()
        true = _spike(len(SPECTRAL_FREQS), 20)
        freqs = torch.tensor(SPECTRAL_FREQS)
        loss = loss_fn(true, true, freqs)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_mass_scale_invariant(self):
        """Same property as SpectralWassersteinLoss: scaling a prediction
        by a constant (adding a constant in log-space) must not change the
        loss — softmax normalises away any uniform additive shift."""
        loss_fn = SpectralKLDivergenceLoss()
        freqs = torch.tensor(SPECTRAL_FREQS)
        true = _spike(len(SPECTRAL_FREQS), 20)
        pred = _spike(len(SPECTRAL_FREQS), 30)

        loss_unscaled = loss_fn(pred, true, freqs).item()
        loss_scaled = loss_fn(pred + np.log(5.0), true, freqs).item()
        assert loss_scaled == pytest.approx(loss_unscaled, rel=1e-4)

    def test_unnormalized_delta_f_matches_normalized(self):
        """Confirms the invariance argument in SpectralKLDivergenceLoss's
        docstring: normalising Δf to sum to 1 (as utils.trapz_weights does)
        only ever subtracts the SAME scalar from every bin's log(Δf), which
        log_softmax is exactly invariant to — so the class's internal,
        unnormalised _trapz_bin_widths must give identical results to a
        manually-normalised version."""
        freqs = torch.tensor(SPECTRAL_FREQS)
        true = _spike(len(SPECTRAL_FREQS), 20)
        pred = _spike(len(SPECTRAL_FREQS), 24)

        loss_fn = SpectralKLDivergenceLoss()
        unnormalized_result = loss_fn(pred, true, freqs).item()

        df = _trapz_bin_widths(freqs)
        log_df_n = torch.log(df / df.sum())
        logp_pred = torch.log_softmax(pred + log_df_n, dim=-1)
        logp_true = torch.log_softmax(true + log_df_n, dim=-1)
        p_true = torch.exp(logp_true)
        normalized_result = (p_true * (logp_true - logp_pred)).sum(-1).item()

        assert unnormalized_result == pytest.approx(normalized_result, rel=1e-5)

    def test_respects_nonuniform_grid(self):
        """A bump in a narrow bin vs. the same-height bump in a wide bin
        (on a deliberately uneven grid) must give different loss — a pure
        index-based (non-Δf-aware) computation would be insensitive to
        this, same sanity check as SpectralWassersteinLoss's version."""
        loss_fn = SpectralKLDivergenceLoss()
        freqs = torch.tensor([0.02, 0.03, 0.07, 0.08, 0.20, 0.485])
        flat_true = torch.log(torch.full((6,), 1.0))
        assert loss_fn(flat_true, flat_true, freqs).item() == pytest.approx(0.0, abs=1e-5)

        pred_a = torch.log(torch.tensor([1.0, 1.0, 2.0, 1.0, 1.0, 1.0]))  # bump, narrow bin
        pred_b = torch.log(torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0, 1.0]))  # bump, wide bin
        loss_a = loss_fn(pred_a, flat_true, freqs).item()
        loss_b = loss_fn(pred_b, flat_true, freqs).item()
        assert loss_a != pytest.approx(loss_b, rel=1e-3)

    def test_batched_multistep_shape(self):
        """(batch, lead_time, num_freqs) — the shape train_one_epoch's
        y_pred/y_batch actually have — must work."""
        loss_fn = SpectralKLDivergenceLoss()
        freqs = torch.tensor(SPECTRAL_FREQS)
        batch, lead_time = 3, 4
        true_1d = _spike(len(SPECTRAL_FREQS), 20)
        true = true_1d.expand(batch, lead_time, -1)
        pred = true.clone()

        loss = loss_fn(pred, true, freqs)
        assert loss.dim() == 0
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_reduction_none_matches_reduction_mean(self):
        loss_fn = SpectralKLDivergenceLoss()
        freqs = torch.tensor(SPECTRAL_FREQS)
        n = len(SPECTRAL_FREQS)
        true = torch.stack([_spike(n, 15), _spike(n, 20), _spike(n, 30)])
        pred = torch.stack([_spike(n, 16), _spike(n, 20), _spike(n, 35)])

        per_sample = loss_fn(pred, true, freqs, reduction='none')
        scalar = loss_fn(pred, true, freqs, reduction='mean')

        assert per_sample.shape == (3,)
        assert per_sample.mean().item() == pytest.approx(scalar.item(), rel=1e-5)
        # Identical pair (index 1) must be exactly zero; the others must not.
        assert per_sample[1].item() == pytest.approx(0.0, abs=1e-5)
        assert per_sample[0].item() > 0.0
        assert per_sample[2].item() > 0.0

    def test_reduction_per_bin_sums_to_none(self):
        """per_bin's per-frequency summand must sum (not integrate — KL's
        bins are already a discrete pmf, unlike Wasserstein's CDF) to the
        'none' reduction's per-sample value."""
        loss_fn = SpectralKLDivergenceLoss()
        freqs = torch.tensor(SPECTRAL_FREQS)
        n = len(SPECTRAL_FREQS)
        true = torch.stack([_spike(n, 15), _spike(n, 30)])
        pred = torch.stack([_spike(n, 16), _spike(n, 35)])

        per_bin = loss_fn(pred, true, freqs, reduction='per_bin')
        per_sample = loss_fn(pred, true, freqs, reduction='none')

        assert per_bin.shape == (2, n)
        assert torch.allclose(per_bin.sum(dim=-1), per_sample, atol=1e-5)

    def test_reduction_invalid_raises(self):
        loss_fn = SpectralKLDivergenceLoss()
        freqs = torch.tensor(SPECTRAL_FREQS)
        true = _spike(len(SPECTRAL_FREQS), 20)
        with pytest.raises(ValueError, match="reduction"):
            loss_fn(true, true, freqs, reduction='sum')

    def test_penalizes_true_peak_loss_far_more_than_modest_hallucination(self):
        """The asymmetric/mode-covering property of KL divergence: a
        MODEST secondary bump elsewhere that stays clearly subordinate to
        (non-competitive with) the true peak's height barely raises the
        loss above 0, while genuinely losing probability mass AT the true
        peak's location (here, via a full shift) raises it
        catastrophically. (A "same-magnitude-either-direction" framing was
        tried and rejected: because probabilities must sum to 1, moving
        mass INTO one bin necessarily moves it OUT of others, so a naive
        symmetric swap does not cleanly isolate this property — comparing
        both losses directly against the shared zero baseline does.)"""
        loss_fn = SpectralKLDivergenceLoss()
        freqs = torch.tensor(SPECTRAL_FREQS)
        n = len(SPECTRAL_FREQS)
        true = _spike(n, 30, height=10.0)

        # Modest hallucination: secondary bump well below the true peak's
        # height (non-competitive in softmax terms) — true peak's own
        # predicted probability is barely disturbed.
        pred_modest_hallucination = _spike(n, 30, height=10.0)
        pred_modest_hallucination[5] = np.log(5.0)
        kl_hallucination = loss_fn(pred_modest_hallucination, true, freqs).item()

        # Full loss of true-location mass: peak moved entirely elsewhere.
        pred_full_shift = _spike(n, 45, height=10.0)
        kl_shift = loss_fn(pred_full_shift, true, freqs).item()

        assert kl_hallucination > 0.0  # still costs something
        assert kl_shift > 20 * kl_hallucination  # but vastly less than losing the peak


class TestSpectralKLDivergenceLossVsLogSpaceMSE:
    """The specific motivating comparison: a peak that BROADENS/partially
    collapses into its immediate neighbourhood (while still leaving real
    mass at the true bin) vs. a peak that fully SHIFTS elsewhere. Plain
    log-space MSE penalises broadening MORE than a full shift (it sums
    squared error across every newly-elevated neighbouring bin); KL
    penalises broadening LESS than a full shift, since its dominant term
    only cares about probability retained exactly at the true bin. KL does
    NOT inherit SpectralWassersteinLoss's shift-tolerance — the shift case
    is still expensive under KL, just less so than under KL's own
    treatment of broadening."""

    def test_discounts_local_broadening_relative_to_shift_unlike_logspace_mse(self):
        loss_fn = SpectralKLDivergenceLoss()
        freqs = torch.tensor(SPECTRAL_FREQS)
        n = len(SPECTRAL_FREQS)
        true = _spike(n, 30, height=10.0)

        pred_shift = _spike(n, 40, height=10.0)  # full 10-bin shift

        broad = np.full(n, 1e-3)
        broad[28:33] = [4.0, 6.0, 7.0, 6.0, 4.0]  # local plateau around bin 30
        pred_broad = torch.log(torch.tensor(broad, dtype=torch.float32))

        ce_shift = loss_fn(pred_shift, true, freqs).item()
        ce_broad = loss_fn(pred_broad, true, freqs).item()

        def logspace_mse(pred, true):
            return ((pred - true) ** 2).mean().item()

        mse_shift = logspace_mse(pred_shift, true)
        mse_broad = logspace_mse(pred_broad, true)

        # CE discounts broadening relative to a full shift...
        assert ce_broad < ce_shift
        # ...while log-space MSE does the OPPOSITE: it penalises broadening
        # even more than a full shift (every elevated neighbouring bin adds
        # its own squared-error term).
        assert mse_broad > mse_shift
