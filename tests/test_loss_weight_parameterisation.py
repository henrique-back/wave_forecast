"""
Tests for the v14 composite-loss parameterisation — nn/optimization.py's
LOSS_TERM_REFERENCE / resolve_loss_weights, and nn/training_loop.py's per-term
loss_components breakdown.

Both exist because train_one_epoch sums four terms raw while they live in four
unrelated units spanning ~4.5 orders of magnitude, so a raw weight says nothing
about how much its term contributes. v13 sampled the three weights
independently and ended up with a space whose MEDIAN draw was ~350:1
peak-dominated — the regime decision 032 showed is degenerate for the
position-blind peak term — without that ever being visible in a log line.
See manuscript/decisions/log/032 and 033.
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nn.optimization import LOSS_TERM_REFERENCE, resolve_loss_weights
from nn.training_loop import train_one_epoch


class TestResolveLossWeights:
    def test_v14_params_convert_contributions_into_weights(self):
        """weight = contribution / reference, so that weight * reference is the
        contribution asked for."""
        params = {"kl_contrib": 1.0, "w2_rel": 2.0, "peak_rel": 0.5}
        w = resolve_loss_weights(params)
        assert w["kl_loss_weight"] == pytest.approx(1.0 / LOSS_TERM_REFERENCE["kl"])
        assert w["wasserstein_loss_weight"] == pytest.approx(2.0 / LOSS_TERM_REFERENCE["w2"])
        assert w["peak_loss_weight"] == pytest.approx(0.5 / LOSS_TERM_REFERENCE["peak"])

    def test_rel_params_are_contribution_ratios_against_kl(self):
        """The whole point of anchoring on KL: `*_rel` must come back out as
        the ratio of realised contributions, not of raw weights."""
        params = {"kl_contrib": 0.3, "w2_rel": 4.0, "peak_rel": 7.0}
        w = resolve_loss_weights(params)
        kl_contribution = w["kl_loss_weight"] * LOSS_TERM_REFERENCE["kl"]
        w2_contribution = w["wasserstein_loss_weight"] * LOSS_TERM_REFERENCE["w2"]
        peak_contribution = w["peak_loss_weight"] * LOSS_TERM_REFERENCE["peak"]
        assert w2_contribution / kl_contribution == pytest.approx(4.0)
        assert peak_contribution / kl_contribution == pytest.approx(7.0)

    def test_pre_v14_params_pass_through_unchanged(self):
        """scripts/train.py must still retrain a v13 study from its saved
        best_trial.txt, which stored absolute weights."""
        params = {"kl_loss_weight": 19.86, "wasserstein_loss_weight": 76.6,
                  "peak_loss_weight": 0.0746, "seq_len": 12}
        assert resolve_loss_weights(params) == {
            "kl_loss_weight": 19.86, "wasserstein_loss_weight": 76.6,
            "peak_loss_weight": 0.0746}

    def test_neither_parameterisation_raises_rather_than_zeroing(self):
        """The pre-v14 call site used params.get(key, 0.0), so renaming a
        parameter silently produced an all-zero loss that surfaced much later
        as a misleading 'predates v13' error. Fail at the lookup instead."""
        with pytest.raises(KeyError):
            resolve_loss_weights({"seq_len": 12, "batch_size": 64})


class TestSearchSpaceIsCentred:
    """The defect being fixed: v13's independent ranges put the median draw far
    into the peak-dominated regime. These pin the new space's shape."""

    V14_SPACE = {"kl_contrib": (0.05, 5.0), "w2_rel": (0.02, 20.0), "peak_rel": (0.02, 20.0)}

    def _sample(self, rng, n=2000):
        """Log-uniform draws, matching trial.suggest_float(..., log=True)."""
        return {k: np.exp(rng.uniform(np.log(lo), np.log(hi), n))
                for k, (lo, hi) in self.V14_SPACE.items()}

    def test_contribution_imbalance_is_bounded_in_both_directions(self):
        draws = self._sample(np.random.default_rng(0))
        for rel in (draws["peak_rel"], draws["w2_rel"]):
            assert rel.max() <= 20.0 and rel.min() >= 0.02

    def test_median_draw_is_near_balance_not_peak_dominated(self):
        """v13's median draw gave the peak term ~350x KL's contribution. The
        median `peak_rel` here must be O(1)."""
        draws = self._sample(np.random.default_rng(1))
        assert 0.3 < float(np.median(draws["peak_rel"])) < 3.0

    def test_no_term_can_be_switched_fully_off(self):
        """A floor above zero lets TPE find 'this term isn't needed' without
        reaching exactly zero — which for the peak term is the degenerate
        direction (decision 032)."""
        assert self.V14_SPACE["peak_rel"][0] > 0.0
        assert self.V14_SPACE["w2_rel"][0] > 0.0

    def test_known_good_ablation_point_is_representable(self):
        """lossablation_v3's 'peak' arm (kl=19.86, peak=0.0746) realised a
        peak:KL contribution ratio of ~11.7 and behaved well, so the ceiling
        has to sit above it rather than below."""
        kl_contribution = 19.86 * LOSS_TERM_REFERENCE["kl"]
        peak_contribution = 0.0746 * LOSS_TERM_REFERENCE["peak"]
        observed_rel = peak_contribution / kl_contribution
        assert observed_rel == pytest.approx(11.7, rel=0.1)
        lo, hi = self.V14_SPACE["peak_rel"]
        assert lo < observed_rel < hi

    def test_v13_space_would_have_been_wildly_imbalanced(self):
        """Regression guard on the diagnosis itself: recomputing v13's bounds
        through the measured references must still reproduce the ~350:1 median
        and the ~6e5:1 worst case that motivated this change."""
        kl_med = np.sqrt(0.1 * 50.0) * LOSS_TERM_REFERENCE["kl"]
        peak_med = np.sqrt(0.01 * 20.0) * LOSS_TERM_REFERENCE["peak"]
        assert peak_med / kl_med > 100.0
        worst = (20.0 * LOSS_TERM_REFERENCE["peak"]) / (0.1 * LOSS_TERM_REFERENCE["kl"])
        assert worst > 1e5


class TestTrainOneEpochLossComponents:
    """train_one_epoch's breakdown — the thing whose absence hid all of the
    above through 240 v13 trials."""

    def _run(self, **weights):
        """One epoch on the same tiny synthetic setup tests/test_training_loop.py
        uses — reused rather than duplicated so both suites exercise one model
        and one data generator."""
        from tests.test_training_loop import _make_shape_model, _make_loader
        torch.manual_seed(0)
        model, _, num_freqs = _make_shape_model()
        loader, freqs_t, shape_means = _make_loader(num_freqs, batch=4, seq_len=5, lead_time=3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return train_one_epoch(model, loader, optimizer, freqs=freqs_t,
                               freq_means=torch.ones(num_freqs), shape_means=shape_means,
                               **weights)

    def test_components_sum_to_the_reported_total(self):
        metrics = self._run(base_loss_weight=0.0, kl_loss_weight=5.0, peak_loss_weight=2.0)
        components = metrics["loss_components"]
        assert sum(components.values()) == pytest.approx(metrics["RMSE"], rel=1e-5)

    def test_inactive_terms_record_zero_not_nan(self):
        metrics = self._run(base_loss_weight=1.0)
        components = metrics["loss_components"]
        assert components["kl"] == 0.0
        assert components["w2"] == 0.0
        assert components["peak"] == 0.0
        assert components["base"] > 0.0

    def test_shares_expose_a_dominated_run(self):
        """A weight 1000x out of balance must show up as a share near 100%,
        which is the signal that was missing."""
        metrics = self._run(base_loss_weight=0.0, kl_loss_weight=1e-3, peak_loss_weight=10.0)
        components = metrics["loss_components"]
        total = sum(abs(v) for v in components.values())
        assert components["peak"] / total > 0.95
