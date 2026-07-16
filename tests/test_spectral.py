"""
Tests for spectral metric correctness around the normalize → denormalize
round-trip.

Key invariant being checked:
    compute_hs_from_density(E_phys, freqs)  ≈  Hs_target          (< 2 % error)
    compute_hs_from_density(E_norm, freqs)  ≠  Hs_target           (> 2 % error)
    compute_hs_from_density(E_norm * μ, freqs)  ≈  Hs_target       (< 2 % error)

where E_norm = E_phys / μ  and  μ is a per-frequency training mean that
differs from E_phys.

The JONSWAP spectrum is parameterised so that 4√m₀ = Hs exactly on the
discrete frequency grid (using the same np.trapezoid call that the code
uses), so the target Hs is met by construction and any residual error is
purely from the normalise/denormalise round-trip.
"""

import sys
import os

import numpy as np
import pytest

# Allow imports from the project root regardless of where pytest is invoked
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.compute_hs import compute_hs_from_density, compute_bulk_params, compute_shape


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jonswap(freqs: np.ndarray, Hs: float, Tp: float, gamma: float = 3.3) -> np.ndarray:
    """Parametric JONSWAP spectrum scaled so that 4√m₀ = Hs exactly.

    The scaling factor α is chosen analytically:
        α = (Hs/4)² / ∫shape(f) df
    so that the trapezoidal integral on `freqs` returns (Hs/4)² by
    construction — the only numerical error is floating-point rounding.
    """
    fp = 1.0 / Tp
    sigma = np.where(freqs <= fp, 0.07, 0.09)
    r = np.exp(-((freqs - fp) ** 2) / (2.0 * sigma**2 * fp**2))
    shape = freqs ** (-5) * np.exp(-1.25 * (fp / freqs) ** 4) * (gamma**r)
    m0_shape = np.trapezoid(shape, freqs)
    alpha = (Hs / 4.0) ** 2 / m0_shape
    return (alpha * shape).astype(np.float32)


# Log-spaced frequency grid matching typical directional wave buoy output
# (0.03 – 0.50 Hz, 64 bins).  Log spacing means Δf is NOT uniform — using
# a hardcoded scalar df would give wrong moment integrals.
FREQS = np.logspace(np.log10(0.03), np.log10(0.50), 64).astype(np.float32)


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


class TestJONSWAPRoundTrip:
    """Normalise → denormalise must preserve Hs within 2 %."""

    @pytest.mark.parametrize(
        "Hs, Tp",
        [
            (1.0, 8.0),  # calm swell-dominated sea
            (2.0, 10.0),  # moderate mixed sea
            (4.0, 14.0),  # energetic long-period swell
        ],
    )
    def test_hs_physical_spectrum_matches_target(self, Hs, Tp):
        """4√m₀ of a correctly constructed JONSWAP equals the target Hs."""
        E = _jonswap(FREQS, Hs, Tp)
        Hs_calc = compute_hs_from_density(E[np.newaxis, :], FREQS)[0]
        rel_err = abs(Hs_calc - Hs) / Hs
        assert rel_err < 0.02, (
            f"Hs={Hs}m Tp={Tp}s: physical-spectrum Hs error {rel_err:.4%} ≥ 2 %"
        )

    @pytest.mark.parametrize(
        "Hs, Tp",
        [
            (1.0, 8.0),
            (2.0, 10.0),
            (4.0, 14.0),
        ],
    )
    def test_hs_after_roundtrip(self, Hs, Tp):
        """Denormalised spectrum recovers Hs to within 2 %."""
        E_phys = _jonswap(FREQS, Hs, Tp)

        # Simulate a per-frequency training mean from a different sea state
        # so that normalisation is non-trivial (μ ≠ E_phys).
        freq_means = _jonswap(FREQS, Hs * 0.6, Tp * 0.85)
        freq_means = np.clip(freq_means, 1e-8, None)

        E_norm = E_phys / freq_means  # normalise
        E_rec = E_norm * freq_means  # denormalise — must recover E_phys

        Hs_orig = compute_hs_from_density(E_phys[np.newaxis, :], FREQS)[0]
        Hs_rec = compute_hs_from_density(E_rec[np.newaxis, :], FREQS)[0]

        rel_err = abs(Hs_rec - Hs_orig) / Hs_orig
        assert rel_err < 0.02, (
            f"Hs={Hs}m Tp={Tp}s: round-trip relative error {rel_err:.4%} ≥ 2 %"
        )

    @pytest.mark.parametrize(
        "Hs, Tp",
        [
            (1.0, 8.0),
            (2.0, 10.0),
        ],
    )
    def test_normalised_spectrum_gives_wrong_hs(self, Hs, Tp):
        """Calling compute_hs on a normalised spectrum (without denormalising)
        produces a value that differs from the physical Hs by > 2 %.

        This test documents the pre-fix bug: passing Ẽ instead of E to
        compute_hs_from_density gives incorrect results when μ ≠ E_phys.
        """
        E_phys = _jonswap(FREQS, Hs, Tp)

        # Use a materially different training mean to guarantee a > 2 % error
        freq_means = _jonswap(FREQS, Hs * 3.0, Tp * 0.65)
        freq_means = np.clip(freq_means, 1e-8, None)

        E_norm = E_phys / freq_means

        Hs_physical = compute_hs_from_density(E_phys[np.newaxis, :], FREQS)[0]
        Hs_from_norm = compute_hs_from_density(E_norm[np.newaxis, :], FREQS)[0]

        rel_err = abs(Hs_from_norm - Hs_physical) / Hs_physical
        assert rel_err > 0.02, (
            "Expected compute_hs on a normalised spectrum to produce > 2 % "
            f"error vs physical Hs, but got {rel_err:.4%}.  Check that "
            "freq_means is sufficiently different from E_phys."
        )


# ---------------------------------------------------------------------------
# Bulk parameter round-trip (Hs and Tm02 via compute_bulk_params)
# ---------------------------------------------------------------------------


class TestBulkParamsRoundTrip:
    """compute_bulk_params must return physically correct Hs and Tm02 when
    called on denormalised spectra."""

    @pytest.mark.parametrize(
        "Hs, Tp",
        [
            (2.0, 10.0),
            (3.5, 12.0),
        ],
    )
    def test_bulk_params_after_roundtrip(self, Hs, Tp):
        """After normalise → denormalise, Hs error < 2 % and Tm02 error < 2 %."""
        E_phys = _jonswap(FREQS, Hs, Tp)
        freq_means = _jonswap(FREQS, Hs * 0.7, Tp * 0.9)
        freq_means = np.clip(freq_means, 1e-8, None)

        E_rec = (E_phys / freq_means) * freq_means  # round-trip

        # compute_bulk_params expects (batch_or_time, num_freqs) or
        # (batch, lead_time, num_freqs); use 2-D form here.
        hs_orig, tm02_orig = compute_bulk_params(E_phys[np.newaxis, :], FREQS)
        hs_rec, tm02_rec = compute_bulk_params(E_rec[np.newaxis, :], FREQS)

        hs_err = abs(hs_rec[0] - hs_orig[0]) / max(hs_orig[0], 1e-12)
        tm02_err = abs(tm02_rec[0] - tm02_orig[0]) / max(tm02_orig[0], 1e-12)

        assert hs_err < 0.02, f"Hs round-trip error {hs_err:.4%} ≥ 2 %"
        assert tm02_err < 0.02, f"Tm02 round-trip error {tm02_err:.4%} ≥ 2 %"

    @pytest.mark.parametrize(
        "Hs, Tp",
        [
            (2.0, 10.0),
        ],
    )
    def test_bulk_params_on_normalised_gives_wrong_values(self, Hs, Tp):
        """compute_bulk_params on normalised spectra gives wrong Hs and/or Tm02."""
        E_phys = _jonswap(FREQS, Hs, Tp)
        freq_means = _jonswap(FREQS, Hs * 2.5, Tp * 0.6)
        freq_means = np.clip(freq_means, 1e-8, None)

        E_norm = E_phys / freq_means

        hs_phys, tm02_phys = compute_bulk_params(E_phys[np.newaxis, :], FREQS)
        hs_norm, tm02_norm = compute_bulk_params(E_norm[np.newaxis, :], FREQS)

        hs_err = abs(hs_norm[0] - hs_phys[0]) / max(hs_phys[0], 1e-12)
        assert hs_err > 0.02, (
            "Expected compute_bulk_params on normalised spectrum to produce "
            f"> 2 % Hs error, but got {hs_err:.4%}"
        )


# ---------------------------------------------------------------------------
# Shape normalisation (compute_shape unit-area property)
# ---------------------------------------------------------------------------


class TestComputeShapeUnitArea:
    """compute_shape(E, freqs) must integrate to 1 over the frequency grid"""

    @pytest.mark.parametrize(
        "Hs, Tp",
        [
            (1.0, 8.0),
            (2.0, 10.0),
            (4.0, 14.0),
        ],
    )
    def test_shape_integrates_to_one(self, Hs, Tp):
        E = _jonswap(FREQS, Hs, Tp)
        shape = compute_shape(E[np.newaxis, :], FREQS)
        integral = np.trapezoid(shape[0], FREQS)
        assert abs(integral - 1.0) < 1e-4, (
            f"Hs={Hs}m Tp={Tp}s: shape integral {integral:.6f} != 1"
        )

    def test_shape_batched_and_multistep_integrates_to_one(self):
        """Same property holds for the (batch, lead_time, num_freqs) shape
        used by nn/evaluate.py, not just the 2-D (samples, num_freqs) shape."""
        E = np.stack(
            [
                _jonswap(FREQS, 1.5, 9.0),
                _jonswap(FREQS, 3.0, 12.0),
            ]
        )[:, np.newaxis, :].repeat(3, axis=1)  # (batch=2, lead_time=3, num_freqs)

        shape = compute_shape(E, FREQS)
        integrals = np.trapezoid(shape, FREQS, axis=2)  # (batch, lead_time)
        assert np.allclose(integrals, 1.0, atol=1e-4)

    def test_shape_handles_near_zero_energy(self):
        """Near-zero-energy input must not produce inf/nan (m0_threshold clip)."""
        E = np.full((1, len(FREQS)), 1e-20, dtype=np.float32)
        shape = compute_shape(E, FREQS)
        assert np.all(np.isfinite(shape))
