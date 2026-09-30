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
    threshold    : γ* threshold (default 1.0 per Portilla et al. 2009, section 3b,
                   building on Violante-Carvalho et al. 2002)

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

    Every local maximum starts as the peak of a partition bounded by the
    minima to its neighbouring maxima. Four criteria mark a partition as
    spurious:
      1. fp > f_max (0.35–0.4 Hz) — tail noise
      2. partition energy < energy_frac * E_total (5%–8%)
      3. fewer than min_bins spectral bins between the peak and either
         partition limit (trough)
      4. peak sits between two higher-energy neighbors (local sandwich)

    Spurious partitions are combined into a neighbour, not dropped, and
    the criteria are re-checked on the combined partitions — see
    _combined_partitions for the order, which Portilla et al. leave open
    (manuscript/decisions/log/030).

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
    return [idx for idx, _, _ in _combined_partitions(freqs, spectrum, f_max, energy_frac, min_bins)]


def _trough(spectrum: np.ndarray, left_idx: int, right_idx: int) -> int:
    """Index of the minimum between two peaks."""
    segment = spectrum[left_idx:right_idx + 1]
    return left_idx + int(np.argmin(segment))


def _combined_partitions(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    f_max: float,
    energy_frac: float,
    min_bins: int,
) -> list[tuple[int, int, int]]:
    """
    Partition-combining loop behind find_significant_peaks/find_peak_windows.

    Repeatedly: partition the spectrum at the minima between the current
    peaks, check the four criteria, and merge the failing peak with the
    lowest S(fp) into the neighbour across its shallower trough (the higher
    trough value, i.e. the weaker separation). The merged pair keeps the
    higher of the two peaks, so a dominant peak is never given up to a
    ripple beside it. A failing peak with no neighbour is dropped. Stops
    when no peak fails; each pass removes one peak, so it terminates.

    Returns (peak_idx, left_idx, right_idx) per surviving peak, ascending
    frequency. The windows are INCLUSIVE, contiguous (neighbours share their
    trough bin) and run from 0 to len(spectrum)-1.
    """
    from scipy.signal import find_peaks

    peaks = [int(p) for p in find_peaks(spectrum, height=0)[0]]
    E_total = np.trapezoid(spectrum, freqs)
    last = len(spectrum) - 1

    def bounds(i: int) -> tuple[int, int]:
        lo = 0 if i == 0 else _trough(spectrum, peaks[i - 1], peaks[i])
        hi = last if i == len(peaks) - 1 else _trough(spectrum, peaks[i], peaks[i + 1])
        return lo, hi

    def spurious(i: int) -> bool:
        idx = peaks[i]
        lo, hi = bounds(i)
        left_higher = i > 0 and spectrum[peaks[i - 1]] > spectrum[idx]
        right_higher = i < len(peaks) - 1 and spectrum[peaks[i + 1]] > spectrum[idx]
        return (freqs[idx] > f_max                                                          # 1
                or np.trapezoid(spectrum[lo:hi + 1], freqs[lo:hi + 1]) < energy_frac * E_total  # 2
                or idx - lo < min_bins or hi - idx < min_bins                               # 3
                or (left_higher and right_higher))                                          # 4

    while peaks:
        failing = [i for i in range(len(peaks)) if spurious(i)]
        if not failing:
            break
        i = min(failing, key=lambda k: spectrum[peaks[k]])
        neighbours = [k for k in (i - 1, i + 1) if 0 <= k < len(peaks)]
        if not neighbours:
            peaks.pop(i)
            continue
        lo, hi = bounds(i)
        j = max(neighbours, key=lambda k: spectrum[lo] if k < i else spectrum[hi])
        peaks.pop(i if spectrum[peaks[i]] <= spectrum[peaks[j]] else j)

    return [(idx, *bounds(i)) for i, idx in enumerate(peaks)]


def find_peak_windows(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    f_max: float = 0.4,
    energy_frac: float = 0.05,
    min_bins: int = 2,
) -> list[tuple[int, int, int]]:
    """
    Like find_significant_peaks, but also returns each surviving peak's
    trough-to-trough partition window (left_idx, right_idx).

    Motivation: a differentiable "soft peak height" training loss
    (utils.loss.SoftPeakHeightLoss) needs a per-peak window to run a
    softmax over, and the window should be the physically-motivated
    trough-to-trough partition span — narrow for a narrow swell partition,
    wide for a broad wind-sea partition — rather than an arbitrary fixed
    bin-radius around the peak, which would be too wide for one regime or
    too narrow for the other.

    The windows are the COMBINED partitions find_significant_peaks settles
    on (same _combined_partitions call): a spurious partition has been
    merged into its neighbour, so each window runs to the trough of the
    next SURVIVING peak, and together the windows tile the whole grid
    without gaps or overlap (adjacent windows share their trough bin).

    Not differentiable, not batched — this loops in Python over a single
    1-D spectrum via scipy.signal.find_peaks.
    utils.loss.SoftPeakHeightLoss.forward expects left_idx/right_idx already
    computed rather than deriving them itself. The ideal call site is once
    per sample at data-preparation time (mirroring how freq_means/
    shape_means are computed once in nn/optimization.py::
    _prepare_dataloaders); the current training loop instead recomputes
    windows per batch as a deliberate, provisional trade-off — see
    manuscript/decisions/log/025.

    Parameters
    ----------
    freqs, spectrum, f_max, energy_frac, min_bins : same as
        find_significant_peaks.

    Returns
    -------
    list[tuple[int, int, int]] — (peak_idx, left_idx, right_idx) per
    surviving peak, in ascending frequency order. left_idx/right_idx are
    INCLUSIVE bin indices; 0 and len(spectrum)-1 at the spectrum's own
    edges when the peak is the first/last partition.
    """
    return _combined_partitions(freqs, spectrum, f_max, energy_frac, min_bins)

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