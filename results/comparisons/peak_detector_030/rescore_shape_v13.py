"""Re-score shape_v13 peak panel: old vs new detector x shape vs physical labels (decision 030)."""
import sys, json, time, importlib.util
from pathlib import Path
import numpy as np, pandas as pd, torch
root = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(root))
SP = Path(__file__).parent  # old_partitioning.py = git show 4534f47:utils/spectral_partitioning.py
from nn import evaluate
from nn.checkpoints import build_model
from nn.optimization import _prepare_dataloaders, _compute_val_score
from utils import get_freqs
import utils.spectral_peaks as sp
import utils.spectral_partitioning as new_part

spec = importlib.util.spec_from_file_location('old_partitioning', SP / 'old_partitioning.py')
old_part = importlib.util.module_from_spec(spec); spec.loader.exec_module(old_part)
DETECTORS = {'old': old_part, 'new': new_part}

def panel(detector, f, pred, true, m0):
    mod = DETECTORS[detector]
    sp.find_peak_windows, sp.find_significant_peaks = mod.find_peak_windows, mod.find_significant_peaks
    scale = 1.0 if m0 is None else m0[:, None]
    m, mm = sp.peak_modality_metrics(f, pred * scale, true * scale)
    m['peak_fidelity_SS'] = _compute_val_score(m, 'peak_fidelity_SS')
    m['multimodal_frac'] = float(mm.mean())
    return m

density, alpha_1, alpha_2, r_1, r_2, wind = pd.read_pickle(root / 'buoy_data/32012/processed_data.pkl')
freqs = get_freqs(density); f = freqs.numpy().astype(float)
out = {}
for lead_h in (12, 24, 48):
    ckpt = torch.load(root / f'results/shape_v13/shape/lead_{lead_h}h/best_model.pt', map_location='cpu', weights_only=False)
    seq_len, lead = ckpt['params']['seq_len'], ckpt['lead_time_steps']
    _, val_loader, test_loader, fm, sm, *_ = _prepare_dataloaders(
        density, alpha_1, alpha_2, r_1, r_2, seq_len, lead, 256, 'shape', shuffle_seed=0,
        wind=wind, channel_set='full', aux_set='dmd')
    model = build_model(ckpt, freqs, 'cpu', 'full', 'dmd')
    for split, loader in (('val', val_loader), ('test', test_loader)):
        t0 = time.time()
        _, (yp, yt, _) = evaluate(model, loader, 'cpu', freqs, lead_time=lead, freq_means=fm,
                                  shape_means=sm, return_arrays=True, compute_peak_metrics=False)
        pred = np.exp(yp[:, -1].numpy().astype(float)); true = np.exp(yt[:, -1].numpy().astype(float))
        m0 = np.asarray(loader.dataset.m0_true, dtype=float)[:, -1]
        res = {f'{d}_{lab}': panel(d, f, pred, true, None if lab == 'shape' else m0)
               for d in ('old', 'new') for lab in ('shape', 'phys')}
        out[f'lead_{lead_h}h_{split}'] = res
        print(f'lead {lead_h}h {split}: N={len(pred)} ({time.time()-t0:.0f}s)', flush=True)
        (SP / 'rescore_shape_v13.json').write_text(json.dumps(out, indent=1))
