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
                         SpectralKLDivergenceLoss, SoftPeakHeightLoss,
                         _trapz_bin_widths, _FULL_CHANNELS, _cumulative_trapz)
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
    — as of 2026-08-17 this is Wasserstein-2 (quadratic transport cost),
    computed via quantile-function inversion, not the earlier Wasserstein-1
    CDF-L1 shortcut (see the class docstring for why W1's shortcut has no
    p=2 analogue). The properties below are what distinguish it from a
    pointwise loss like RMSELoss: forgiving of small position shifts,
    mass-scale invariant (it compares shape, not magnitude) — both
    properties W1 also had; W2 additionally penalizes one large
    displacement more harshly than several small ones summing to the same
    distance (quadratic vs. linear transport cost)."""

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
        """A narrow peak shifted by a small vs. large number of bins: W2
        must scale smoothly/monotonically with shift distance, while a
        plain pointwise error on the same pairs is roughly CONSTANT (a
        narrow peak has near-total non-overlap even for a 1-bin shift, so
        pointwise error can't distinguish a near miss from a far one) —
        this is the exact property motivating Wasserstein over RMSELoss/the
        (reverted) SpectralSlopeLoss for the multimodal-blur problem."""
        loss_fn = SpectralWassersteinLoss()
        freqs = torch.tensor(SPECTRAL_FREQS)
        n = len(SPECTRAL_FREQS)
        true = _spike(n, 20)

        def pointwise_mse(pred_log):
            pred_n = torch.exp(pred_log) / torch.exp(pred_log).sum()
            true_n = torch.exp(true) / torch.exp(true).sum()
            return ((pred_n - true_n) ** 2).mean().item()

        w2_near = loss_fn(_spike(n, 21), true, freqs).item()   # shift by 1 bin
        w2_mid = loss_fn(_spike(n, 25), true, freqs).item()    # shift by 5 bins
        w2_far = loss_fn(_spike(n, 45), true, freqs).item()    # shift by 25 bins

        assert w2_near < w2_mid < w2_far  # W2 scales with distance

        mse_near = pointwise_mse(_spike(n, 21))
        mse_far = pointwise_mse(_spike(n, 45))
        assert mse_near == pytest.approx(mse_far, rel=0.05)  # pointwise error is blind to distance

    def test_matches_brute_force_quantile_reference(self):
        """Cross-check against an independent NumPy reference: exponentiate,
        mass-normalize, build CDFs via manual cumulative trapz, invert both
        via np.interp on a FINE (5000-point) uniform quantile grid, then
        W2 = sqrt(trapz((quantile_pred-quantile_true)^2, q_grid)). This is
        the same quantile-matching formula forward() uses, computed
        independently and at much higher quadrature resolution — confirms
        the true-anchored 64-point quadrature forward() actually uses
        (see class docstring: true's own CDF values double as the
        quadrature nodes) converges to the same answer as a dedicated fine
        grid, not just that the code runs.

        Uses a genuinely multi-peaked, unevenly-shaped pair (not pure
        point-mass spikes) so the transport plan being integrated isn't a
        degenerate single-displacement case.
        """
        loss_fn = SpectralWassersteinLoss()
        freqs_np = np.asarray(SPECTRAL_FREQS, dtype=np.float64)
        freqs_t = torch.tensor(SPECTRAL_FREQS)
        n = len(SPECTRAL_FREQS)

        true_shape = np.abs(np.sin(np.linspace(0.0, 6.0, n))) + 0.05
        pred_shape = np.abs(np.sin(np.linspace(0.3, 6.3, n))) + 0.08
        true_log = torch.log(torch.tensor(true_shape, dtype=torch.float32))
        pred_log = torch.log(torch.tensor(pred_shape, dtype=torch.float32))

        w2_impl = loss_fn(pred_log, true_log, freqs_t).item()

        def _cum_trapz_np(y, x):
            seg = (y[1:] + y[:-1]) / 2 * (x[1:] - x[:-1])
            return np.concatenate([[0.0], np.cumsum(seg)])

        true_norm = true_shape / np.trapezoid(true_shape, freqs_np)
        pred_norm = pred_shape / np.trapezoid(pred_shape, freqs_np)
        cdf_true = _cum_trapz_np(true_norm, freqs_np)
        cdf_pred = _cum_trapz_np(pred_norm, freqs_np)

        q_grid = np.linspace(1e-6, 1 - 1e-6, 5000)
        quant_true = np.interp(q_grid, cdf_true, freqs_np)
        quant_pred = np.interp(q_grid, cdf_pred, freqs_np)
        w2_ref = float(np.sqrt(np.trapezoid((quant_pred - quant_true) ** 2, q_grid)))

        assert w2_impl == pytest.approx(w2_ref, rel=0.02)

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

    def test_reduction_per_bin_integrates_to_none(self):
        """per_bin's squared-quantile-gap summand, trapz-integrated over
        cdf_true (NOT freqs — see class docstring: per_bin's domain is
        cumulative probability, not frequency) and then sqrt'd, must
        reproduce the 'none' reduction's per-sample W2 exactly — not a
        different computation path."""
        loss_fn = SpectralWassersteinLoss()
        freqs = torch.tensor(SPECTRAL_FREQS)
        n = len(SPECTRAL_FREQS)
        true = torch.stack([_spike(n, 15), _spike(n, 30)])
        pred = torch.stack([_spike(n, 16), _spike(n, 35)])

        per_bin = loss_fn(pred, true, freqs, reduction='per_bin')
        per_sample = loss_fn(pred, true, freqs, reduction='none')

        assert per_bin.shape == (2, n)

        true_phys = torch.exp(true)
        true_norm = true_phys / torch.trapezoid(true_phys, freqs, dim=-1).unsqueeze(-1)
        cdf_true = _cumulative_trapz(true_norm, freqs)

        w2_from_per_bin = torch.sqrt(torch.trapezoid(per_bin, cdf_true, dim=-1).clamp(min=0.0))
        assert torch.allclose(w2_from_per_bin, per_sample, atol=1e-5)


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


def _small_grid():
    """Uniform 6-bin, 1 Hz-step toy grid — the exact shift/collapse worked
    examples from the loss-design discussion, kept deliberately tiny and
    hand-verifiable rather than reusing the 64-bin SPECTRAL_FREQS grid."""
    return torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])


def _to_log(values, floor=1e-3):
    """A plain (non-log) list of physical values -> LOG-space tensor, zeros
    floored (mirrors _spike's floor, and utils.log_transform.to_log_space's
    floor-before-log convention) so torch.log never sees an exact 0.0."""
    x = np.array(values, dtype=np.float32)
    x = np.where(x <= 0, floor, x)
    return torch.log(torch.tensor(x, dtype=torch.float32))


class TestSoftPeakHeightLoss:
    """SoftPeakHeightLoss (utils/loss.py) — the third auxiliary spectral
    loss term, complementary to SpectralWassersteinLoss: Wasserstein
    measures how far energy moved; this measures whether a peak's own
    height survived, independent of position. Uses the exact shift/collapse
    pair worked out by hand in the loss-design discussion (see class docstring), on the
    tiny 6-bin _small_grid() rather than SPECTRAL_FREQS, so expected
    values can be reasoned about directly."""

    def test_shift_gives_identical_residual_to_perfect_prediction(self):
        """A pure position shift doesn't change the MULTISET of values in
        the window, only which bin holds which value — and H_pred depends
        only on that multiset (a softmax-weighted average is invariant to
        a permutation of which index holds which value). So a shifted
        peak must score EXACTLY as well as a perfect (unshifted)
        prediction: both compare the identical soft H_pred value against
        the same hard H_true. This is the precise sense in which this loss
        is 'invariant to pure shift' — an identity, not an approximation."""
        loss_fn = SoftPeakHeightLoss()
        freqs = _small_grid()
        true = _to_log([0, 0.2, 0.6, 0.2, 0, 0])
        pred_shift = _to_log([0, 0, 0.2, 0.6, 0.2, 0])  # same values, shifted by 1 bin
        left_idx = torch.tensor([0])
        right_idx = torch.tensor([5])
        peak_mask = torch.tensor([True])

        perfect = loss_fn(true, true, freqs, left_idx, right_idx, peak_mask).item()
        shift = loss_fn(pred_shift, true, freqs, left_idx, right_idx, peak_mask).item()
        # rel=1e-4, not tighter: float32 softmax/exp isn't bit-identical
        # under a mere reordering of the same summands, only equal to
        # float32 precision — the identity itself is exact in real
        # arithmetic (see docstring above).
        assert shift == pytest.approx(perfect, rel=1e-4)

    def test_perfect_prediction_residual_is_small_but_nonzero(self):
        """H_true is a hard max but H_pred is a soft (temperature-weighted)
        estimate of the SAME values — a soft estimate is generically <=
        the hard max of those values, so even a perfect prediction leaves
        a small positive residual (shrinking as tau_k shrinks). This is an
        intentional trade-off (the label side doesn't need gradient, so
        there's no reason to soften it — see class docstring), not a bug:
        the residual must stay small, not literally zero."""
        loss_fn = SoftPeakHeightLoss()
        freqs = _small_grid()
        true = _to_log([0, 0.2, 0.6, 0.2, 0, 0])
        left_idx = torch.tensor([0])
        right_idx = torch.tensor([5])
        peak_mask = torch.tensor([True])

        perfect = loss_fn(true, true, freqs, left_idx, right_idx, peak_mask).item()
        assert 0.0 < perfect < 0.01

    def test_local_collapse_scores_worse_than_shift_unlike_wasserstein(self):
        """The complementary-blind-spots property motivating this class:
        for the SAME pair of predictions, SpectralWassersteinLoss rates a
        local peak collapse as CHEAPER than a pure shift (it only measures
        how far mass travelled — collapsing to immediate neighbours moves
        less mass, less far, than translating the whole peak) — while
        SoftPeakHeightLoss rates the collapse as clearly WORSE than the
        shift (it destroys the peak's own height, which a pure shift never
        touches — see test_shift_gives_identical_residual_to_perfect_
        prediction)."""
        peak_loss_fn = SoftPeakHeightLoss()
        w1_loss_fn = SpectralWassersteinLoss()
        freqs = _small_grid()
        true = _to_log([0, 0.2, 0.6, 0.2, 0, 0])
        pred_shift = _to_log([0, 0, 0.2, 0.6, 0.2, 0])
        pred_collapse = _to_log([0, 0.4, 0.2, 0.4, 0, 0])
        left_idx = torch.tensor([0])
        right_idx = torch.tensor([5])
        peak_mask = torch.tensor([True])

        peak_shift = peak_loss_fn(pred_shift, true, freqs, left_idx, right_idx, peak_mask).item()
        peak_collapse = peak_loss_fn(pred_collapse, true, freqs, left_idx, right_idx, peak_mask).item()
        w1_shift = w1_loss_fn(pred_shift, true, freqs).item()
        w1_collapse = w1_loss_fn(pred_collapse, true, freqs).item()

        assert w1_collapse < w1_shift            # Wasserstein's blind spot
        assert peak_collapse > 10 * peak_shift   # SoftPeakHeightLoss's correction

    def test_tau_limits_hard_max_and_mean(self):
        """tau -> 0 (forced via tiny c/tau_min) must recover H_pred ==
        max(E_pred window); tau -> infinity (forced via huge tau_min) must
        recover H_pred == mean(E_pred window) — the two degenerate cases
        softmax interpolates between (class docstring)."""
        freqs = _small_grid()
        true = _to_log([0, 0.2, 0.6, 0.2, 0, 0])
        pred = _to_log([0, 0.4, 0.2, 0.4, 0, 0])
        left_idx = torch.tensor([0])
        right_idx = torch.tensor([5])
        peak_mask = torch.tensor([True])

        pred_phys = torch.exp(pred)
        true_max = torch.exp(true).max().item()

        sharp_fn = SoftPeakHeightLoss(c=1e-8, tau_min=1e-8)
        sharp = sharp_fn(pred, true, freqs, left_idx, right_idx, peak_mask,
                          reduction='per_peak').item()
        expected_sharp = (pred_phys.max().item() - true_max) ** 2
        assert sharp == pytest.approx(expected_sharp, rel=1e-3)

        soft_fn = SoftPeakHeightLoss(c=1.0, tau_min=1e6)
        soft = soft_fn(pred, true, freqs, left_idx, right_idx, peak_mask,
                        reduction='per_peak').item()
        expected_soft = (pred_phys.mean().item() - true_max) ** 2
        assert soft == pytest.approx(expected_soft, rel=1e-3)

        assert sharp != pytest.approx(soft, rel=0.1)

    def test_gradient_flows_to_every_bin_in_window_none_outside(self):
        """The core value proposition over max(): every bin actually
        inside the peak's window must receive nonzero gradient (unlike
        argmax, which zeroes out every bin except the single winner),
        while bins OUTSIDE the window (not part of this peak's partition)
        must receive exactly zero gradient."""
        freqs = _small_grid()
        true = _to_log([0, 0.2, 0.6, 0.2, 0, 0])
        pred = _to_log([0, 0.4, 0.2, 0.4, 0, 0]).clone().requires_grad_(True)
        left_idx = torch.tensor([1])   # window covers only bins 1..4
        right_idx = torch.tensor([4])
        peak_mask = torch.tensor([True])

        loss_fn = SoftPeakHeightLoss()
        loss = loss_fn(pred, true, freqs, left_idx, right_idx, peak_mask)
        loss.backward()

        assert pred.grad is not None
        for i in range(1, 5):
            assert pred.grad[i].item() != 0.0
        for i in (0, 5):
            assert pred.grad[i].item() == 0.0

    def test_reduction_per_peak_and_none_consistent_with_mean(self):
        loss_fn = SoftPeakHeightLoss()
        freqs = _small_grid()
        true = _to_log([0, 0.2, 0.6, 0.2, 0, 0])
        pred = _to_log([0, 0.4, 0.2, 0.4, 0, 0])
        left_idx = torch.tensor([0])
        right_idx = torch.tensor([5])
        peak_mask = torch.tensor([True])

        per_peak = loss_fn(pred, true, freqs, left_idx, right_idx, peak_mask, reduction='per_peak')
        none_val = loss_fn(pred, true, freqs, left_idx, right_idx, peak_mask, reduction='none')
        mean_val = loss_fn(pred, true, freqs, left_idx, right_idx, peak_mask, reduction='mean')

        assert per_peak.shape == (1,)
        assert none_val.item() == pytest.approx(per_peak[0].item(), rel=1e-6)
        assert mean_val.item() == pytest.approx(none_val.item(), rel=1e-6)

    def test_k_zero_sample_excluded_from_mean_not_zero_filled(self):
        """A sample with no detected peak at all (peak_mask all False)
        must be EXCLUDED from 'mean', not folded in as a filled-in zero —
        mirrors nn/evaluate.py's M0_MASK_THRESHOLD/valid.any() pattern for
        a quantity undefined for that sample, rather than diluting the
        batch mean toward 'no error' for a sample this loss has nothing to
        say about."""
        loss_fn = SoftPeakHeightLoss()
        freqs = _small_grid()
        true = _to_log([0, 0.2, 0.6, 0.2, 0, 0])
        pred_collapse = _to_log([0, 0.4, 0.2, 0.4, 0, 0])

        true_b = true.unsqueeze(0).expand(2, -1)
        pred_b = pred_collapse.unsqueeze(0).expand(2, -1)
        left_idx = torch.tensor([[0, 0], [0, 0]])
        right_idx = torch.tensor([[5, 5], [5, 5]])
        peak_mask = torch.tensor([[True, False], [False, False]])  # sample 1: K=0

        per_sample = loss_fn(pred_b, true_b, freqs, left_idx, right_idx, peak_mask, reduction='none')
        scalar = loss_fn(pred_b, true_b, freqs, left_idx, right_idx, peak_mask, reduction='mean')

        assert per_sample.shape == (2,)
        assert per_sample[1].item() == pytest.approx(0.0, abs=1e-8)  # K=0 -> safe-divide 0
        assert scalar.item() == pytest.approx(per_sample[0].item(), rel=1e-5)
        assert scalar.item() != pytest.approx(per_sample.mean().item(), rel=1e-3)

    def test_all_samples_k_zero_gives_nan(self):
        loss_fn = SoftPeakHeightLoss()
        freqs = _small_grid()
        true = _to_log([0, 0.2, 0.6, 0.2, 0, 0])
        left_idx = torch.tensor([0])
        right_idx = torch.tensor([5])
        peak_mask = torch.tensor([False])  # no real peak anywhere in the batch

        scalar = loss_fn(true, true, freqs, left_idx, right_idx, peak_mask)
        assert torch.isnan(scalar)

    def test_out_of_range_padding_indices_do_not_crash(self):
        """Padding slots using an out-of-range sentinel (e.g. -1, a common
        convention) must not produce NaN/inf — the defensive clamp in
        forward() must catch this regardless of what convention an
        eventual caller uses."""
        loss_fn = SoftPeakHeightLoss()
        freqs = _small_grid()
        true = _to_log([0, 0.2, 0.6, 0.2, 0, 0])
        pred = _to_log([0, 0.4, 0.2, 0.4, 0, 0])
        left_idx = torch.tensor([0, -1])
        right_idx = torch.tensor([5, -1])
        peak_mask = torch.tensor([True, False])

        scalar = loss_fn(pred, true, freqs, left_idx, right_idx, peak_mask)
        per_peak = loss_fn(pred, true, freqs, left_idx, right_idx, peak_mask, reduction='per_peak')
        assert torch.isfinite(scalar)
        assert torch.isfinite(per_peak).all()

    def test_batched_multistep_shape(self):
        """(batch, lead_time, num_freqs) with (batch, lead_time, max_peaks)
        windows — the shape train_one_epoch's y_pred/y_batch actually
        have — must work, not just a bare 1-D spectrum."""
        loss_fn = SoftPeakHeightLoss()
        freqs = _small_grid()
        true_1d = _to_log([0, 0.2, 0.6, 0.2, 0, 0])
        batch, lead_time, max_peaks = 3, 4, 1
        true = true_1d.expand(batch, lead_time, -1)
        pred = true.clone()
        left_idx = torch.zeros(batch, lead_time, max_peaks, dtype=torch.long)
        right_idx = torch.full((batch, lead_time, max_peaks), 5, dtype=torch.long)
        peak_mask = torch.ones(batch, lead_time, max_peaks, dtype=torch.bool)

        loss = loss_fn(pred, true, freqs, left_idx, right_idx, peak_mask)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_reduction_invalid_raises(self):
        loss_fn = SoftPeakHeightLoss()
        freqs = _small_grid()
        true = _to_log([0, 0.2, 0.6, 0.2, 0, 0])
        left_idx = torch.tensor([0])
        right_idx = torch.tensor([5])
        peak_mask = torch.tensor([True])
        with pytest.raises(ValueError, match="reduction"):
            loss_fn(true, true, freqs, left_idx, right_idx, peak_mask, reduction='sum')
