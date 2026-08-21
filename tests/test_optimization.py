"""
Tests for nn/optimization.py::_compute_val_score's 'peak_fidelity_SS'
objective metric — added 2026-08-19 to fix scripts/ablate_loss.py using an
RMSE-rooted metric (final_step_SS) to pick the best epoch/trial for arms
that don't train on RMSE at all (base_loss_weight=0). See that metric's
docstring entry in _compute_val_score for the full rationale.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nn.optimization import _compute_val_score


def _metrics(windsea_rel_err, swell_rel_err, windsea_recall, swell_recall):
    return {
        'Peak_Height_RelError_windsea': windsea_rel_err,
        'Peak_Height_RelError_swell': swell_rel_err,
        'Peak_Separation_Recall_windsea': windsea_recall,
        'Peak_Separation_Recall_swell': swell_recall,
    }


class TestPeakFidelitySS:
    def test_perfect_prediction_gives_best_possible_score(self):
        """Zero relative error, perfect recall on both labels -> recall(1.0) -
        rel_err(0.0) = 1.0, the maximum this metric can produce."""
        metrics = _metrics(0.0, 0.0, 1.0, 1.0)
        assert _compute_val_score(metrics, 'peak_fidelity_SS') == pytest.approx(1.0)

    def test_higher_is_better_direction(self):
        """Lower relative error and/or higher recall must score strictly
        higher — this is what the LR scheduler (mode='max') and Optuna
        (direction='maximize') both assume."""
        worse = _metrics(0.5, 0.5, 0.5, 0.5)
        better_recall = _metrics(0.5, 0.5, 0.9, 0.9)
        better_relerr = _metrics(0.1, 0.1, 0.5, 0.5)
        base_score = _compute_val_score(worse, 'peak_fidelity_SS')
        assert _compute_val_score(better_recall, 'peak_fidelity_SS') > base_score
        assert _compute_val_score(better_relerr, 'peak_fidelity_SS') > base_score

    def test_one_label_missing_falls_back_to_the_other(self):
        """A validation pass with, say, no true swell partitions detected at
        all (both swell keys NaN together — see peak_modality_metrics)
        must not poison the score; np.nanmean over the two labels should
        just use whichever one is real."""
        metrics = _metrics(0.2, float('nan'), 0.8, float('nan'))
        score = _compute_val_score(metrics, 'peak_fidelity_SS')
        assert score == pytest.approx(0.8 - 0.2)

    def test_both_labels_missing_gives_negative_infinity_not_nan(self):
        """No true peak detected in EITHER label across the whole
        validation pass -- must not propagate NaN into the LR
        scheduler/pruner (a NaN comparison is silently always False,
        which would corrupt early-stopping/pruning decisions)."""
        metrics = _metrics(float('nan'), float('nan'), float('nan'), float('nan'))
        score = _compute_val_score(metrics, 'peak_fidelity_SS')
        assert score == float('-inf')
        assert not np.isnan(score)

    def test_missing_keys_raises_rather_than_silently_falling_back(self):
        """compute_peak_metrics=False upstream (the default -- see
        _train_model) means these keys simply aren't in `metrics`.
        Deliberately a hard KeyError, not a silent fallback to an
        RMSE-rooted metric -- that fallback is exactly the bug this metric
        exists to avoid reintroducing."""
        with pytest.raises(KeyError):
            _compute_val_score({}, 'peak_fidelity_SS')

    def test_does_not_disturb_existing_metric_names(self):
        """Adding the new branch must not change any pre-existing
        objective_metric's behavior (the live shape_v12-style study uses
        'final_step_SS_wasserstein' and must be unaffected)."""
        metrics = {'per_step_SS': [0.1, 0.2, 0.3], 'Shape_Wasserstein': 0.05}
        assert _compute_val_score(metrics, 'final_step_SS') == pytest.approx(0.3)
