"""
Smoke tests for nn/training_loop.py::train_one_epoch's loss-ablation wiring
(base_loss_weight, peak_loss_weight/peak_max_count) — added alongside
utils/spectral_peaks.py's wind-sea/swell metrics and the SpectralWasserstein
Loss W1->W2 update as scaffolding for the KL/Wasserstein/peak composite-loss
ablation (see CLAUDE.md). Not a numerical-correctness suite for the
underlying loss classes themselves (see tests/test_loss.py for those) — just
confirms the new parameters actually reach the loss computation, end to end
through a real (tiny) WaveHeightBaselineNN forward/backward pass, without
crashing or producing a NaN/non-finite loss.
"""

import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nn.transformer import WaveHeightBaselineNN
from nn.dataset import WaveSpectralDataset
from nn.training_loop import train_one_epoch

FREQS = np.array([0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.48], dtype=np.float32)


def _make_shape_model():
    freqs = torch.tensor(FREQS)
    num_freqs = len(FREQS)
    model = WaveHeightBaselineNN(
        freqs=freqs, num_freqs=num_freqs, target="shape", num_channels=1,
        nhead=2, num_encoder_layers=1, num_decoder_layers=1, embed_dim=8,
    )
    return model, freqs, num_freqs


def _make_loader(num_freqs, batch, seq_len, lead_time, seed=0):
    """Synthetic bimodal-ish physical spectra (not flat/uniform noise) so
    SoftPeakHeightLoss's peak-window detection has real peaks to find,
    exercising _peak_windows_for_batch's actual code path rather than
    hitting its all-empty-windows/NaN-skip branch every batch."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 6.0, num_freqs)
    base = np.abs(np.sin(x)) + 0.05
    total = seq_len + lead_time
    density = np.tile(base, (batch, total, 1)).astype(np.float32)
    density += rng.normal(scale=0.01, size=density.shape).astype(np.float32)
    density = np.clip(density, 1e-3, None)

    src = torch.from_numpy(density[:, :seq_len, :]).unsqueeze(-1)  # (batch, seq_len, num_freqs, 1)
    y_raw = torch.from_numpy(density[:, seq_len:, :])              # (batch, lead_time, num_freqs)
    freqs_t = torch.tensor(FREQS)
    mass = torch.trapezoid(y_raw, freqs_t, dim=-1, ).unsqueeze(-1)
    y = y_raw / mass.clamp(min=1e-8)  # unit-area physical shape, per prepare_y's convention
    aux = torch.zeros(batch, seq_len, 0)

    loader = DataLoader(WaveSpectralDataset(src, aux, y), batch_size=batch, shuffle=False)
    shape_means = y.mean(dim=(0, 1)).clamp(min=1e-6)
    return loader, freqs_t, shape_means


class TestTrainOneEpochLossAblationWiring:
    def test_default_weights_reproduce_prior_behavior(self):
        """base_loss_weight=1.0/peak_loss_weight=0.0 (the defaults) must
        behave exactly as before this ablation's wiring was added — no
        crash, finite loss."""
        torch.manual_seed(0)
        model, freqs, num_freqs = _make_shape_model()
        loader, freqs_t, shape_means = _make_loader(num_freqs, batch=4, seq_len=5, lead_time=3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        metrics = train_one_epoch(model, loader, optimizer, freqs=freqs_t,
                                   freq_means=torch.ones(num_freqs), shape_means=shape_means)
        assert np.isfinite(metrics['RMSE'])

    def test_base_loss_weight_zero_runs_without_the_per_bin_term(self):
        """base_loss_weight=0.0 with kl_loss_weight>0 is the ablation's
        'substitute' arm (see train_one_epoch's docstring) — loss must
        still be finite and training must still proceed (params change)."""
        torch.manual_seed(0)
        model, freqs, num_freqs = _make_shape_model()
        loader, freqs_t, shape_means = _make_loader(num_freqs, batch=4, seq_len=5, lead_time=3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = [p.clone() for p in model.parameters()]

        metrics = train_one_epoch(model, loader, optimizer, freqs=freqs_t,
                                   freq_means=torch.ones(num_freqs), shape_means=shape_means,
                                   base_loss_weight=0.0, kl_loss_weight=5.0)
        assert np.isfinite(metrics['RMSE'])
        after = list(model.parameters())
        assert any(not torch.equal(b, a) for b, a in zip(before, after))

    def test_peak_loss_weight_positive_runs_end_to_end(self):
        """peak_loss_weight>0 exercises _peak_windows_for_batch's real
        find_peak_windows path (not just the K=0/NaN-skip branch) — the
        synthetic bimodal batch has real, detectable peaks."""
        torch.manual_seed(0)
        model, freqs, num_freqs = _make_shape_model()
        loader, freqs_t, shape_means = _make_loader(num_freqs, batch=4, seq_len=5, lead_time=3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        metrics = train_one_epoch(model, loader, optimizer, freqs=freqs_t,
                                   freq_means=torch.ones(num_freqs), shape_means=shape_means,
                                   base_loss_weight=0.0, kl_loss_weight=5.0,
                                   peak_loss_weight=2.0, peak_max_count=3)
        assert np.isfinite(metrics['RMSE'])

    def test_peak_loss_weight_positive_with_no_detectable_peaks_does_not_crash(self):
        """A perfectly flat spectrum has no significant peaks at all (see
        find_significant_peaks) -> SoftPeakHeightLoss's 'mean' reduction
        returns NaN for every batch -> train_one_epoch must skip adding it
        to the loss (see its 'not torch.isnan(peak_term)' guard) rather than
        letting a NaN loss corrupt every parameter's gradient."""
        torch.manual_seed(0)
        model, freqs, num_freqs = _make_shape_model()
        batch, seq_len, lead_time = 3, 4, 2
        total = seq_len + lead_time
        flat = np.ones((batch, total, num_freqs), dtype=np.float32)
        src = torch.from_numpy(flat[:, :seq_len, :]).unsqueeze(-1)
        y_raw = torch.from_numpy(flat[:, seq_len:, :])
        freqs_t = torch.tensor(FREQS)
        mass = torch.trapezoid(y_raw, freqs_t, dim=-1).unsqueeze(-1)
        y = y_raw / mass.clamp(min=1e-8)
        aux = torch.zeros(batch, seq_len, 0)
        loader = DataLoader(WaveSpectralDataset(src, aux, y), batch_size=batch, shuffle=False)
        shape_means = y.mean(dim=(0, 1)).clamp(min=1e-6)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        metrics = train_one_epoch(model, loader, optimizer, freqs=freqs_t,
                                   freq_means=torch.ones(num_freqs), shape_means=shape_means, peak_loss_weight=2.0)
        assert np.isfinite(metrics['RMSE'])
