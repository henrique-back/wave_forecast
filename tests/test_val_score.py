"""
Tests for nn/optimization.py::_compute_val_score — in particular the
'final_step_SS_wasserstein' objective_metric added so that model/checkpoint
selection is structurally consistent with the training loss once it includes
an auxiliary Wasserstein term (wasserstein_loss_weight), rather than
selecting by a metric (plain final_step_SS) that's blind to the exact
peak-separation quality that term exists to improve.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nn.optimization import _compute_val_score, _FINAL_STEP_SS_WASSERSTEIN_BETA


def _make_metrics(per_step_ss, shape_wasserstein=None, **extra):
    m = {'per_step_SS': per_step_ss, 'overall_SS': 0.2, 'Hs_SS': 0.15,
         'RMSE': 1.5, 'Hs_RMSE': 0.3, 'Tm02_RMSE': 0.4, 'Shape_RMSE': 2.0,
         'SI_mean': 0.5}
    if shape_wasserstein is not None:
        m['Shape_Wasserstein'] = shape_wasserstein
    m.update(extra)
    return m


class TestExistingBranchesUnaffected:
    """Regression guard: adding the new branch must not disturb the others."""

    def test_final_step_ss(self):
        m = _make_metrics([0.1, 0.12, 0.15])
        assert _compute_val_score(m, 'final_step_SS') == pytest.approx(0.15)

    def test_shape_rmse_is_negated(self):
        m = _make_metrics([0.1])
        assert _compute_val_score(m, 'Shape_RMSE') == pytest.approx(-2.0)

    def test_unknown_metric_raises(self):
        m = _make_metrics([0.1])
        with pytest.raises(ValueError, match="final_step_SS_wasserstein"):
            _compute_val_score(m, 'not_a_real_metric')


class TestFinalStepSSWasserstein:
    def test_matches_manual_formula(self):
        m = _make_metrics([0.05, 0.10, 0.18], shape_wasserstein=0.012)
        expected = 0.18 - _FINAL_STEP_SS_WASSERSTEIN_BETA * 0.012
        assert _compute_val_score(m, 'final_step_SS_wasserstein') == pytest.approx(expected)

    def test_higher_wasserstein_distance_lowers_score(self):
        """Shape_Wasserstein is a distance (lower is better) — increasing it
        at fixed final_step_SS must lower the combined score."""
        m_good = _make_metrics([0.15], shape_wasserstein=0.010)
        m_bad = _make_metrics([0.15], shape_wasserstein=0.020)
        score_good = _compute_val_score(m_good, 'final_step_SS_wasserstein')
        score_bad = _compute_val_score(m_bad, 'final_step_SS_wasserstein')
        assert score_good > score_bad

    def test_does_not_depend_on_any_trial_specific_weight(self):
        """The blend weight must be a fixed module constant, not read from
        the metrics dict — otherwise cross-trial comparison would be unfair
        (see _FINAL_STEP_SS_WASSERSTEIN_BETA's docstring). Injecting an
        unrelated 'wasserstein_loss_weight' key into metrics must not change
        the score."""
        m = _make_metrics([0.15], shape_wasserstein=0.012,
                           wasserstein_loss_weight=999.0)
        expected = 0.15 - _FINAL_STEP_SS_WASSERSTEIN_BETA * 0.012
        assert _compute_val_score(m, 'final_step_SS_wasserstein') == pytest.approx(expected)
