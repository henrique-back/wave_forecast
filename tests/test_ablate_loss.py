"""
Tests for scripts/ablate_loss.py::_fixed_weights_for_phase — the per-phase
loss-recipe assembly the whole ablation's arm structure depends on. Only
covers phases that don't touch disk (baseline, kl, wasserstein_only,
peak_only): wasserstein/peak/combined call _read_prior_weight, which reads
a real completed 'kl' phase's current_best.txt — appropriately validated
by this project's actual completed runs rather than a mocked-file unit
test here.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.ablate_loss import _fixed_weights_for_phase


class TestFixedWeightsForPhase:
    def test_baseline_is_the_unperturbed_production_loss(self):
        base_loss_weight, fixed = _fixed_weights_for_phase("baseline")
        assert base_loss_weight == 1.0
        assert fixed == {"kl_loss_weight": 0.0, "wasserstein_loss_weight": 0.0,
                          "peak_loss_weight": 0.0}

    def test_kl_substitutes_with_everything_else_zero(self):
        base_loss_weight, fixed = _fixed_weights_for_phase("kl")
        assert base_loss_weight == 0.0
        assert fixed == {"wasserstein_loss_weight": 0.0, "peak_loss_weight": 0.0}
        assert "kl_loss_weight" not in fixed  # that's the one THIS phase searches

    def test_wasserstein_only_has_no_kl_and_no_disk_dependency(self):
        """The whole point of this phase: kl_loss_weight pinned to 0, not
        read from a prior 'kl' run — must not raise even if no 'kl' phase
        has ever completed."""
        base_loss_weight, fixed = _fixed_weights_for_phase("wasserstein_only")
        assert base_loss_weight == 0.0
        assert fixed == {"kl_loss_weight": 0.0, "peak_loss_weight": 0.0}

    def test_peak_only_has_no_kl_and_no_disk_dependency(self):
        base_loss_weight, fixed = _fixed_weights_for_phase("peak_only")
        assert base_loss_weight == 0.0
        assert fixed == {"kl_loss_weight": 0.0, "wasserstein_loss_weight": 0.0}

    def test_unknown_phase_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Unknown phase"):
            _fixed_weights_for_phase("not_a_real_phase")
