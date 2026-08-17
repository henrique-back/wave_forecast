import numpy as np

# ── JONSWAP spectrum ──────────────────────────────────────────────────────────

def jonswap_spectrum(
    freqs: np.ndarray,
    fp: float,
    alpha: float = 0.0081,
    gamma: float = 3.3,
    sigma_a: float = 0.07,
    sigma_b: float = 0.09,
    g: float = 9.81,
) -> np.ndarray:
    """
    JONSWAP spectrum (Hasselmann et al. 1973).

    Parameters
    ----------
    freqs   : frequency array [Hz]
    fp      : peak frequency [Hz]
    alpha   : Phillips constant (default = PM value 0.0081)
    gamma   : peak enhancement factor (default = 3.3, JONSWAP mean)
    sigma_a : spectral width below peak (default 0.07)
    sigma_b : spectral width above peak (default 0.09)
    g       : gravitational acceleration [m/s²]

    Returns
    -------
    S(f) [m² Hz⁻¹]
    """
    sigma = np.where(freqs < fp, sigma_a, sigma_b)

    # Phillips equilibrium tail
    phillips = alpha * g**2 * (2 * np.pi) ** -4 * freqs**-5

    # Exponential decay toward peak
    decay = np.exp(-1.25 * (fp / freqs) ** 4)

    # Peak enhancement (Gaussian in frequency)
    r = np.exp(-((freqs - fp) ** 2) / (2 * sigma**2 * fp**2))
    enhancement = gamma**r

    return phillips * decay * enhancement


def pm_at_peak(fp: float, alpha_pm: float = 0.0081, g: float = 9.81) -> float:
    """
    PM reference energy evaluated analytically at the peak frequency.
    Eq. (6) with gamma=1, f=fp, alpha=alpha_PM.

    S_PM(fp) = alpha_PM * g² / (2π)⁴ * fp⁻⁵ * exp(-5/4)
    """
    return alpha_pm * g**2 * (2 * np.pi) ** -4 * fp**-5 * np.exp(-1.25)


# ── 1-D identification algorithm ──────────────────────────────────────────────

def gamma_star(S_obs_at_fp: float, fp: float, **pm_kwargs) -> float:
    """
    γ* = S_obs(fp) / S_PM(fp)

    Parameters
    ----------
    S_obs_at_fp : observed spectral energy at the peak frequency [m² Hz⁻¹]
    fp          : peak frequency of the wave system [Hz]

    Returns
    -------
    γ* (dimensionless)
    """
    return S_obs_at_fp / pm_at_peak(fp, **pm_kwargs)


def classify_partition(
    fp: float,
    S_obs_at_fp: float,
    threshold: float = 1.0,
) -> str:
    """
    Classify a single spectral partition as 'wind_sea' or 'swell'.

    Parameters
    ----------
    fp           : peak frequency of the partition [Hz]
    S_obs_at_fp  : spectral energy density at fp [m² Hz⁻¹]
    threshold    : γ* threshold (default 1.0 per Violante-Carvalho 2009)

    Returns
    -------
    'wind_sea' or 'swell'
    """
    gstar = gamma_star(S_obs_at_fp, fp)
    return "wind_sea" if gstar > threshold else "swell"


def classify_partitions(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    peaks: list[int],
    threshold: float = 1.0,
) -> list[dict]:
    """
    Classify a list of spectral peaks (partition indices) from a 1-D spectrum.

    Parameters
    ----------
    freqs    : frequency array [Hz], shape (N,)
    spectrum : 1-D energy spectrum S(f) [m² Hz⁻¹], shape (N,)
    peaks    : list of frequency-array indices corresponding to each partition peak
    threshold: γ* threshold

    Returns
    -------
    List of dicts with keys: fp, S_at_fp, gamma_star, label
    """
    results = []
    for idx in peaks:
        fp = freqs[idx]
        S_fp = spectrum[idx]
        gstar = gamma_star(S_fp, fp)
        results.append(
            {
                "fp": fp,
                "S_at_fp": S_fp,
                "gamma_star": gstar,
                "label": "wind_sea" if gstar > threshold else "swell",
            }
        )
    return results

def find_significant_peaks(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    f_max: float = 0.4,       # criterion 1: upper frequency cutoff [Hz]
    energy_frac: float = 0.05, # criterion 2: min fraction of total energy
    min_bins: int = 2,         # criterion 3: min bins on each side of peak
) -> list[int]:
    """
    Portilla et al. (2009) 1D spurious peak removal (section 2b.2).

    Four criteria mark a peak as spurious:
      1. fp > f_max (0.35–0.4 Hz) — tail noise
      2. partition energy < energy_frac * E_total (5%–8%)
      3. fewer than min_bins spectral bins before or after the peak
      4. peak sits between two higher-energy neighbors (local sandwich)

    Parameters
    ----------
    freqs    : frequency array [Hz]
    spectrum : 1-D energy density S(f) [m² Hz⁻¹]
    f_max    : high-frequency cutoff for criterion 1
    energy_frac : fractional energy threshold for criterion 2
    min_bins : minimum number of bins on either side of peak for criterion 3

    Returns
    -------
    List of indices of significant peaks.
    """
    from scipy.signal import find_peaks

    # All local maxima (raw)
    raw_peaks, _ = find_peaks(spectrum, height=0)
    if len(raw_peaks) == 0:
        return []

    # Partition limits: minima between consecutive peaks
    # For each peak, partition spans from the preceding minimum to the next
    def partition_energy(idx: int, all_peaks: np.ndarray) -> float:
        pos = np.searchsorted(all_peaks, idx)
        lo = 0 if pos == 0 else _trough(spectrum, all_peaks[pos - 1], idx)
        hi = len(spectrum) - 1 if pos == len(all_peaks) - 1 else _trough(spectrum, idx, all_peaks[pos + 1])
        return np.trapezoid(spectrum[lo:hi + 1], freqs[lo:hi + 1])

    E_total = np.trapezoid(spectrum, freqs)

    significant = []
    for i, idx in enumerate(raw_peaks):
        fp = freqs[idx]

        # Criterion 1: high-frequency tail
        if fp > f_max:
            continue

        # Criterion 2: low partition energy
        if partition_energy(idx, raw_peaks) < energy_frac * E_total:
            continue

        # Criterion 3: too few bins on either side of peak
        left_bins = idx - (0 if i == 0 else raw_peaks[i - 1])
        right_bins = (len(spectrum) - 1 if i == len(raw_peaks) - 1 else raw_peaks[i + 1]) - idx
        if left_bins < min_bins or right_bins < min_bins:
            continue

        # Criterion 4: sandwiched between two higher-energy neighbors
        left_higher = i > 0 and spectrum[raw_peaks[i - 1]] > spectrum[idx]
        right_higher = i < len(raw_peaks) - 1 and spectrum[raw_peaks[i + 1]] > spectrum[idx]
        if left_higher and right_higher:
            continue

        significant.append(idx)

    return significant


def _trough(spectrum: np.ndarray, left_idx: int, right_idx: int) -> int:
    """Index of the minimum between two peaks."""
    segment = spectrum[left_idx:right_idx + 1]
    return left_idx + int(np.argmin(segment))


def find_peak_windows(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    f_max: float = 0.4,
    energy_frac: float = 0.05,
    min_bins: int = 2,
) -> list[tuple[int, int, int]]:
    """
    Like find_significant_peaks, but also returns each surviving peak's
    trough-to-trough partition window (left_idx, right_idx) — boundary
    information find_significant_peaks already computes internally (the
    'lo'/'hi' locals inside its partition_energy closure, used only to
    gate criterion 2) but never returns.

    Motivation: a differentiable "soft peak height" training loss
    (utils.loss.SoftPeakHeightLoss) needs a per-peak window to run a
    softmax over, and the window should be the physically-motivated
    trough-to-trough partition span — narrow for a narrow swell partition,
    wide for a broad wind-sea partition — rather than an arbitrary fixed
    bin-radius around the peak, which would be too wide for one regime or
    too narrow for the other. Reusing the partition boundaries this module
    already computes (rather than re-deriving a different notion of
    "window") also guarantees two adjacent peaks' windows are contiguous,
    never overlapping: both share the trough between them as their common
    boundary.

    left_idx/right_idx use EXACTLY the same derivation as
    find_significant_peaks' internal partition_energy (same _trough calls
    against the full raw-peak sequence, not just the surviving peaks) — so
    a window returned here is numerically identical to the span
    find_significant_peaks already used, internally, to decide whether
    that peak passed criterion 2.

    Not differentiable, not batched — like find_significant_peaks, this
    calls scipy.signal.find_peaks and loops in Python over a single 1-D
    spectrum. Call this ONCE per sample at data-preparation time (see
    nn/optimization.py::_prepare_dataloaders' freq_means/shape_means
    precedent for precompute-once-per-run tensors), never inside the
    training hot path — utils.loss.SoftPeakHeightLoss.forward expects
    left_idx/right_idx already computed, exactly as
    nn/evaluate.py:53-60 documents peak-detection being opt-in/
    evaluation-only for the same performance reason.

    Parameters
    ----------
    freqs, spectrum, f_max, energy_frac, min_bins : same as
        find_significant_peaks.

    Returns
    -------
    list[tuple[int, int, int]] — (peak_idx, left_idx, right_idx) per
    surviving peak, in ascending frequency order. left_idx/right_idx are
    INCLUSIVE bin indices; 0 and len(spectrum)-1 at the spectrum's own
    edges when the peak is the first/last partition (same edge convention
    as find_significant_peaks' internal partition_energy).
    """
    from scipy.signal import find_peaks

    raw_peaks, _ = find_peaks(spectrum, height=0)
    if len(raw_peaks) == 0:
        return []

    significant_idx = find_significant_peaks(
        freqs, spectrum, f_max=f_max, energy_frac=energy_frac, min_bins=min_bins
    )
    if not significant_idx:
        return []
    sig_set = set(significant_idx)

    windows = []
    for i, idx in enumerate(raw_peaks):
        if idx not in sig_set:
            continue
        lo = 0 if i == 0 else _trough(spectrum, raw_peaks[i - 1], idx)
        hi = len(spectrum) - 1 if i == len(raw_peaks) - 1 else _trough(spectrum, idx, raw_peaks[i + 1])
        windows.append((int(idx), int(lo), int(hi)))
    return windows

# ── Quick demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import pandas as pd
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from utils import get_freqs


    # ── Load data ─────────────────────────────────────────────────────────────
    BUOY_ID = "32012"
    project_root = Path(__file__).resolve().parent.parent
    folder_path = project_root / "buoy_data" / BUOY_ID
    file_path = folder_path / "processed_data.pkl"

    if file_path.exists():
        dfs_interpolated = pd.read_pickle(file_path)
        density, alpha_1, alpha_2, r_1, r_2, wind = dfs_interpolated
        print("Loaded preprocessed wave spectral data")
    else:
        from utils.data_processing import data_processing
        density, alpha_1, alpha_2, r_1, r_2, wind = data_processing(
            folder_path, save_path=file_path
        )

    freqs = get_freqs(density)  # shape (N_freqs,)

    # ── Classify a sample of timestamps ──────────────────────────────────────
    N_SAMPLES = 5
    sample_times = density.index[:N_SAMPLES]

    for t in sample_times:
        spectrum = density.loc[t].values  # S(f) at this timestamp

        peak_idxs = find_significant_peaks(freqs, spectrum, energy_frac=0.08)
        if len(peak_idxs) == 0:
            print(f"\n{t}: no peaks found")
            continue

        partitions = classify_partitions(freqs, spectrum, peak_idxs)

        print(f"\n{t}")
        for p in partitions:
            print(
                f"  fp={p['fp']:.3f} Hz | γ*={p['gamma_star']:.3f} | {p['label']}"
            )