"""
Lead-time study, step 1 of 2: roll every model out to MAX_STEP hours from one
fixed set of test start times and store per-sample peak-fidelity statistics
(utils/peak_records.py) at every scored step. scripts/plot_lead_curves.py
turns the stored statistics into the figures and tables; nothing there
re-runs a model or a peak detector.

Models ("forecasters"; each returns physical spectra for steps 1..MAX_STEP):
  persistence   the spectrum at the start time T, for every step
  ar            ridge AR (utils/linear_baseline.py) from results/<ar-name>/shape/
                lead_<L>h/linear_baseline_final.pt, one per trained lead L,
                rolled recursively to MAX_STEP. Leads whose selected order is
                the same give identical forecasts; each is still stored.
  gefs          GEFSv12 control member (scripts/fetch_gefs_reforecast.py, must
                reach MAX_STEP), 00Z start times only
  transformer   results/<experiment>/shape/lead_<L>h/final_model_seed*.pt (or
                best_model.pt when no seed checkpoints exist), rolled to
                MAX_STEP with model.infer(). Decoder steps past L were never
                trained; that extrapolation is what the extension figure shows.

Method, fixed for both phases (AR first, transformer later) so adding the
transformer never changes the other models' numbers:
  * Start times: every test row T with WINDOW hours of test-split history
    before it (WINDOW = the largest seq_len / AR order either model can pick,
    so every model can forecast from every T) and T + MAX_STEP still inside
    the test split. The 00Z subset with a GEFS cycle is flagged.
  * Band and units: as scripts/compare_physical_baseline.py. Buoy bins inside
    the GEFS grid (0.0375-0.485 Hz); every spectrum is clipped at 0, cut to
    the band, renormalised to unit area, then scaled by the true band m0 at
    the valid time so the wind-sea/swell labels are physical (decision 029).
    Truth is the raw buoy density, without evaluate()'s floor.
  * Scored steps: STEPS (every 3 h, GEFS's output interval).
  * Frequency bands for the band breakdown: BAND_EDGES_HZ, fixed here before
    any score is looked at.

Output (--out, default results/comparisons/lead_curves/):
  truth.npz           issue_times (S,), steps, band freqs and edges,
                      n_true_peaks / dominant_is_swell / hs (S, n_steps) at
                      the valid time, gefs_rows (indices of the 00Z subset)
  models/<key>.npz    stats (n_rows, n_steps, 1 + n_bands, N_STATS) float32,
                      rows (indices into truth's sample axis), plus metadata
  Existing model files are skipped unless --overwrite, so a run can be
  extended (e.g. transformer after AR) or resumed.

Usage:
    python scripts/eval_lead_curves.py --models persistence ar gefs --ar-name linear_baseline_pf
    python scripts/eval_lead_curves.py --models transformer --experiment shape_v14 --device cuda
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from torch.utils.data import DataLoader, Subset

from utils import get_freqs, compute_shape
from utils.linear_baseline import forecast_coeffs
from utils.peak_records import peak_stats

BUOY_ID = "32012"
CHANNEL_SET = "full"
AUX_SET = "dmd"
MAX_STEP = 96
WINDOW = 96
STEPS = np.arange(3, MAX_STEP + 1, 3)
BAND_EDGES_HZ = (0.08, 0.125, 0.2)
GEFS_RANGE_HZ = (0.035, 0.963)          # GEFSv12 wave grid; checked against the file when loaded
DEFAULT_LEADS = [6, 12, 24, 48, 72, 96]

project_root = Path(__file__).resolve().parent.parent


class Context:
    """Data, start rows and truth shared by every forecaster."""

    def __init__(self, gefs):
        data = pd.read_pickle(project_root / "buoy_data" / BUOY_ID / "processed_data.pkl")
        self.data = data
        density = data[0]
        self.index = density.index
        self.raw = density.to_numpy(dtype=np.float64)
        self.freqs = np.array([float(c) for c in density.columns])
        self.freqs_t = get_freqs(density)
        if not np.allclose(self.freqs, self.freqs_t.numpy()):
            raise ValueError("pickle column order does not match get_freqs")
        self.val_end = int(0.85 * len(density))      # nn/optimization.py::_prepare_dataloaders

        n = len(density)
        self.k = np.arange(self.val_end + WINDOW - 1, n - MAX_STEP)
        self.issue_times = self.index[self.k]
        self.band = (self.freqs >= GEFS_RANGE_HZ[0]) & (self.freqs <= GEFS_RANGE_HZ[1])
        self.fb = self.freqs[self.band]

        # Truth at every valid time T + h for h = 1..MAX_STEP: (S, MAX_STEP, Fb)
        valid = self.k[:, None] + np.arange(1, MAX_STEP + 1)[None, :]
        self.truth = np.clip(self.raw[valid][..., self.band], 0.0, None)
        self.m0 = np.trapezoid(self.truth, self.fb, axis=-1)            # (S, MAX_STEP)

        self.gefs = gefs
        self.gefs_rows = np.array([], dtype=int)
        if gefs is not None:
            gf = gefs["freqs"].astype(np.float64)
            if not np.array_equal(self.band, (self.freqs >= gf.min()) & (self.freqs <= gf.max())):
                raise ValueError("GEFS frequency range differs from GEFS_RANGE_HZ")
            if gefs["lead_hours"].max() < MAX_STEP:
                raise ValueError(f"GEFS file stops at {gefs['lead_hours'].max():.0f} h; "
                                 f"re-run scripts/fetch_gefs_reforecast.py with MAX_LEAD_H >= {MAX_STEP}")
            init = pd.DatetimeIndex(gefs["init_times"])
            pos = pd.Index(self.issue_times).get_indexer(init)
            li = [int(np.flatnonzero(gefs["lead_hours"] == h)[0]) for h in STEPS]
            finite = np.isfinite(gefs["E1d"][:, li]).all(axis=(1, 2))
            keep = (pos >= 0) & finite
            self.gefs_rows = pos[keep]
            self.gefs_cycles = np.flatnonzero(keep)

    def band_scaled(self, spectra, rows):
        """(n, MAX_STEP, F or Fb) physical forecasts -> band unit-area shape x true band m0.
        Full-grid input is cut to the band first."""
        if spectra.shape[-1] == len(self.freqs):
            spectra = spectra[..., self.band]
        shape = compute_shape(np.clip(spectra, 0.0, None), self.fb)
        return shape * self.m0[rows][..., None]


def score(ctx, forecasts, rows, n_jobs):
    """Peak statistics at every scored step. forecasts: (n, MAX_STEP, Fb), already band_scaled."""
    def one(h):
        return peak_stats(ctx.fb, forecasts[:, h - 1], ctx.truth[rows, h - 1], band_edges=BAND_EDGES_HZ)
    out = Parallel(n_jobs=n_jobs)(delayed(one)(h) for h in STEPS)
    stats = np.stack([o[0] for o in out], axis=1).astype(np.float32)     # (n, n_steps, G, N_STATS)
    n_true = np.stack([o[1] for o in out], axis=1)
    dominant = np.stack([o[2] for o in out], axis=1)
    return stats, n_true, dominant


def save_model(out_dir, key, stats, rows, meta):
    path = out_dir / "models" / f"{key}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, stats=stats, rows=rows, meta=json.dumps(meta))
    print(f"  wrote {path} ({len(rows)} start times)")


# ---------------------------------------------------------------------------
# Forecasters: each yields (key, forecasts (n, MAX_STEP, F), rows, meta)
# ---------------------------------------------------------------------------

def persistence(ctx, args):
    rows = np.arange(len(ctx.k))
    fc = np.repeat(ctx.raw[ctx.k][:, None, :], MAX_STEP, axis=1)
    yield "persistence", fc, rows, {"model": "persistence"}


def ar(ctx, args):
    rows = np.arange(len(ctx.k))
    density = ctx.data[0]
    for lead in args.leads:
        path = project_root / "results" / args.ar_name / "shape" / f"lead_{lead}h" / "linear_baseline_final.pt"
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        order = ckpt["order"]
        pred, true, _ = forecast_coeffs(density, ckpt["freqs"], ckpt["coeffs"], order, MAX_STEP,
                                        "shape", eval_split="test")
        i = ctx.k - ctx.val_end - order + 1
        expected = compute_shape(ctx.raw[ctx.k + MAX_STEP], ctx.freqs)
        if not np.allclose(true[i, -1], expected, rtol=1e-4, atol=1e-9):
            raise AssertionError(f"AR lead {lead}: targets are not the buoy spectra at T + h (alignment)")
        yield f"ar_lead{lead}", pred[i], rows, {
            "model": "ar", "name": args.ar_name, "trained_lead": lead, "order": order,
            "ridge": ckpt["ridge"], "checkpoint": str(path.relative_to(project_root)),
            "samples_with_negative_bins": int((pred[i] < 0).any(axis=(1, 2)).sum())}


def gefs(ctx, args):
    if ctx.gefs is None:
        raise FileNotFoundError("GEFS file not found; run scripts/fetch_gefs_reforecast.py")
    g = ctx.gefs
    gf = g["freqs"].astype(np.float64)
    # GEFS is 3-hourly. Fill an hourly (n, MAX_STEP, Fb) array; only STEPS are ever read.
    fc = np.full((len(ctx.gefs_rows), MAX_STEP, len(ctx.fb)), np.nan)
    for h in STEPS:
        (li,) = np.flatnonzero(g["lead_hours"] == h)
        e = g["E1d"][ctx.gefs_cycles, li].astype(np.float64)
        fc[:, h - 1] = np.stack([np.interp(ctx.fb, gf, row) for row in e])
    yield "gefs", fc, ctx.gefs_rows, {"model": "gefs", "member": "c00"}


def transformer(ctx, args):
    from nn.checkpoints import build_model
    from nn.optimization import _prepare_dataloaders

    density, alpha_1, alpha_2, r_1, r_2, wind = ctx.data
    rows = np.arange(len(ctx.k))
    for lead in args.leads:
        folder = project_root / "results" / args.experiment / "shape" / f"lead_{lead}h"
        paths = sorted(folder.glob("final_model_seed*.pt")) or [folder / "best_model.pt"]
        loaders = {}
        for path in paths:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if ckpt["lead_time_steps"] != lead:
                raise ValueError(f"{path}: trained lead {ckpt['lead_time_steps']} != {lead}")
            seq_len = ckpt["params"]["seq_len"]
            if seq_len > WINDOW:
                raise ValueError(f"{path}: seq_len {seq_len} > WINDOW {WINDOW}")
            if seq_len not in loaders:
                # Test windows built for a MAX_STEP target; the input window of
                # sample j is the same as for the trained lead (checked below).
                long_ = _prepare_dataloaders(density, alpha_1, alpha_2, r_1, r_2, seq_len, MAX_STEP, 256,
                                             "shape", shuffle_seed=0, wind=wind,
                                             channel_set=CHANNEL_SET, aux_set=AUX_SET)
                short = _prepare_dataloaders(density, alpha_1, alpha_2, r_1, r_2, seq_len, lead, 256,
                                             "shape", shuffle_seed=0, wind=wind,
                                             channel_set=CHANNEL_SET, aux_set=AUX_SET)
                ds_long, ds_short = long_[2].dataset, short[2].dataset
                m = len(ds_long)
                if not (torch.equal(ds_long.X, ds_short.X[:m]) and torch.equal(ds_long.aux, ds_short.aux[:m])):
                    raise AssertionError(f"seq_len {seq_len}: inputs depend on the target length")
                loaders[seq_len] = (ds_long, long_[3], long_[4])
            ds, freq_means, shape_means = loaders[seq_len]
            if not (torch.allclose(freq_means, ckpt["freq_means"]) and torch.allclose(shape_means, ckpt["shape_means"])):
                raise ValueError(f"{path}: loader normalisation differs from the checkpoint's")
            seed = ckpt.get("seed", "best")
            key = f"transformer_{args.experiment}_lead{lead}_seed{seed}"
            if args.done(key):
                print(f"  {key}: exists, skipped")
                continue
            j = ctx.k - ctx.val_end - seq_len + 1
            expected = compute_shape(ctx.raw[ctx.k + MAX_STEP], ctx.freqs)
            if not np.allclose(ds.y[j, -1].numpy(), expected, rtol=1e-4, atol=1e-7):
                raise AssertionError(f"{path}: targets are not the buoy spectra at T + h (alignment)")

            model = build_model(ckpt, ctx.freqs_t, args.device, CHANNEL_SET, AUX_SET)
            preds = []
            with torch.no_grad():
                for src, aux, _ in DataLoader(Subset(ds, j.tolist()), batch_size=args.batch_size):
                    out = model.infer(src.to(args.device), ctx.freqs_t, MAX_STEP, freq_means=freq_means,
                                      shape_means=shape_means, aux=aux.to(args.device))
                    preds.append(torch.exp(out).cpu().double().numpy())
            yield key, np.concatenate(preds), rows, {
                "model": "transformer", "experiment": args.experiment, "trained_lead": lead,
                "seed": seed, "seq_len": seq_len, "checkpoint": str(path.relative_to(project_root))}


FORECASTERS = {"persistence": persistence, "ar": ar, "gefs": gefs, "transformer": transformer}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["persistence", "ar", "gefs"], choices=list(FORECASTERS))
    p.add_argument("--leads", type=int, nargs="+", default=DEFAULT_LEADS, help="trained leads (ar, transformer)")
    p.add_argument("--ar-name", default="linear_baseline_pf")
    p.add_argument("--experiment", default="shape_v14")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--out", type=Path, default=project_root / "results" / "comparisons" / "lead_curves")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.n_jobs)        # torch otherwise takes every core of a shared host
    gefs_path = project_root / "buoy_data" / BUOY_ID / "gefsv12_c00_spec1d.npz"
    gefs_data = dict(np.load(gefs_path)) if gefs_path.exists() else None
    ctx = Context(gefs_data)
    args.out.mkdir(parents=True, exist_ok=True)
    args.done = lambda key: (args.out / "models" / f"{key}.npz").exists() and not args.overwrite
    print(f"{len(ctx.k)} start times {ctx.issue_times[0]} .. {ctx.issue_times[-1]}; "
          f"{len(ctx.gefs_rows)} with a GEFS cycle; band {ctx.fb[0]:.4f}-{ctx.fb[-1]:.4f} Hz ({ctx.band.sum()} bins)")

    truth_path = args.out / "truth.npz"
    if not truth_path.exists() or args.overwrite:
        # Truth-side peak counts/labels at every scored step (model-independent).
        rows = np.arange(len(ctx.k))
        _, n_true, dominant = score(ctx, ctx.truth, rows, args.n_jobs)
        np.savez_compressed(
            truth_path, issue_times=ctx.issue_times.to_numpy(), steps=STEPS, freqs_band=ctx.fb,
            band_edges=np.array(BAND_EDGES_HZ), n_true_peaks=n_true, dominant_is_swell=dominant,
            hs=4 * np.sqrt(ctx.m0[:, STEPS - 1]), gefs_rows=ctx.gefs_rows,
            meta=json.dumps({"buoy": BUOY_ID, "max_step": MAX_STEP, "window": WINDOW,
                             "band_hz": [float(ctx.fb[0]), float(ctx.fb[-1])]}))
        print(f"  wrote {truth_path}")

    for name in args.models:
        print(f"== {name}")
        for key, fc, rows, meta in FORECASTERS[name](ctx, args):
            if args.done(key):
                print(f"  {key}: exists, skipped")
                continue
            stats, _, _ = score(ctx, ctx.band_scaled(fc, rows), rows, args.n_jobs)
            save_model(args.out, key, stats, rows, meta)


if __name__ == "__main__":
    main()
