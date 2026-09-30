"""Detector statistics on the buoy 32012 test-split truth: old (4534f47) vs new detector (decision 030)."""
import sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
here = Path(__file__).parent; root = here.parents[2]; sys.path.insert(0, str(root))
from utils import get_freqs
import utils.spectral_partitioning as new
spec = importlib.util.spec_from_file_location('old', here / 'old_partitioning.py')
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)

density = pd.read_pickle(root / 'buoy_data/32012/processed_data.pkl')[0]
freqs = get_freqs(density); test = density.iloc[int(0.85 * len(density)):].values
lines = [f'buoy 32012 test split, raw hourly spectra: N = {len(test)}, {len(freqs)} bins']
for name, mod in (('old', old), ('new', new)):
    n, w, g = [], [], 0
    for s in test:
        ws = mod.find_peak_windows(freqs, s)
        n.append(len(ws)); w += [r - l for _, l, r in ws]; g += any(p == int(np.argmax(s)) for p, _, _ in ws)
    n = np.array(n)
    lines.append(f'{name}: peaks/spectrum {n.mean():.2f}, count dist {np.bincount(n).tolist()}, '
                 f'>=2 peaks {(n >= 2).mean():.1%}, >4 peaks {(n > 4).mean():.1%}, '
                 f'median window {np.median(w):.0f} bins, global max kept {g}/{len(test)}')
(here / 'truth_detector_stats.txt').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))
