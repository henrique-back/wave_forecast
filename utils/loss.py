import torch

# Column order DirectionalLoss expects. Deliberately hardcoded rather than
# imported from nn.channels.CHANNEL_SETS['full'] — nn/__init__.py eagerly
# imports submodules (training_loop, optimization) that import from utils,
# so `from nn.channels import ...` here would create a circular import
# (utils -> nn -> utils, mid-initialisation). utils has no dependency on nn
# anywhere else in this codebase; keep it that way. Must stay in sync with
# CHANNEL_SETS['full'] in nn/channels.py.
_FULL_CHANNELS = ['density', 'alpha_1_sin', 'alpha_1_cos',
                   'alpha_2_sin', 'alpha_2_cos', 'r_1', 'r_2']
_DENSITY_IDX      = _FULL_CHANNELS.index('density')
_ALPHA1_SIN_IDX   = _FULL_CHANNELS.index('alpha_1_sin')
_ALPHA1_COS_IDX   = _FULL_CHANNELS.index('alpha_1_cos')
_ALPHA2_SIN_IDX   = _FULL_CHANNELS.index('alpha_2_sin')
_ALPHA2_COS_IDX   = _FULL_CHANNELS.index('alpha_2_cos')
_R1_IDX           = _FULL_CHANNELS.index('r_1')
_R2_IDX           = _FULL_CHANNELS.index('r_2')


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, weights=None, squared=False):
        """
        Parameters
        ----------
        weights : torch.Tensor | None, shape (last_dim,)
            Per-element weights broadcast against the last axis (e.g.
            utils.trapz_weights(freqs), which sums to 1). When given, the
            squared error is summed (not averaged) across the last axis
            using these weights before averaging over all other axes — a
            frequency-weighted MSE instead of a flat elementwise mean.
            When None, behaves exactly like plain MSE (equal weight per
            element), matching the previous nn.MSELoss-based behaviour.
        squared : bool
            When True, return the (possibly frequency-weighted) MSE without
            the final sqrt — i.e. plain MSE, for targets trained directly in
            log-space (see nn/training_loop.py) where the RMSE's extra
            1/(2*RMSE) gradient scaling isn't wanted. Default False preserves
            this class's original RMSE behaviour.
        """
        sq_err = (y_pred - y_true) ** 2
        if weights is None:
            mse = sq_err.mean()
        else:
            mse = (sq_err * weights).sum(dim=-1).mean()
        return mse if squared else torch.sqrt(mse)


def _cumulative_trapz(y, freqs):
    """Cumulative trapezoidal integral of y over freqs, along the last axis.

    Returns a tensor the same shape as y, where result[..., i] = the
    trapezoidal integral of y from freqs[0] to freqs[i] (result[..., 0] == 0).
    PyTorch has no built-in cumulative trapezoid (only whole-integral
    torch.trapezoid), so this is a small manual implementation: cumsum of
    each segment's trapezoid area, zero-prepended.
    """
    delta_f = freqs[1:] - freqs[:-1]  # (num_freqs-1,)
    segment_areas = (y[..., 1:] + y[..., :-1]) / 2 * delta_f  # (..., num_freqs-1)
    cum = torch.cumsum(segment_areas, dim=-1)
    zeros = torch.zeros_like(cum[..., :1])
    return torch.cat([zeros, cum], dim=-1)  # (..., num_freqs)


class SpectralWassersteinLoss(torch.nn.Module):
    """
    1-D Wasserstein-1 (earth-mover) distance between predicted and true
    spectra, treated as probability distributions over frequency — an
    alternative to RMSELoss/SpectralSlopeLoss aimed at the same multimodal
    blurring problem, but with a different, complementary property: W1 is
    naturally forgiving of small position/phase shifts (a peak one bin off
    costs a small, smoothly-scaling penalty) while still penalizing "flat
    blur instead of two spikes" (moving mass from a spike to a spread-out
    blob costs real transport distance, proportional to how far the mass
    moved) — unlike a pointwise loss (RMSELoss, or SpectralSlopeLoss's
    derivative variant), which penalizes a slightly shifted sharp peak
    almost as harshly as a completely displaced one, since a shift produces
    near-zero pointwise/derivative overlap at the peak location.

    For 1-D distributions, W1 has an exact closed form: the L1 distance
    between CDFs (∫|CDF_pred(f) - CDF_true(f)| df) — no optimal-transport
    solver or learned critic network needed (that machinery, e.g. a WGAN's
    critic, is only required to APPROXIMATE Wasserstein distance in high
    dimensions; here the spectrum is a single 1-D curve over an ordered
    frequency axis, so it's computed exactly and cheaply).

    Each spectrum is normalized by its own total mass before building its
    CDF — this compares pure SHAPE, decoupled from magnitude, by design:
    mass conservation is already tracked separately (Shape_Mass_Error, and
    model.infer()'s explicit renormalization), so a systematically
    over/under-scaled prediction should not inflate this loss/metric on its
    own — only genuine distributional (shape) mismatch should.

    This internal mass-normalization is also why the class works unchanged
    for target == 'density': its log-spectral-energy is exponentiated and
    normalized the same way 'shape's log-shape is, so nothing here is
    actually shape-specific — only the (separate) gating in
    nn/training_loop.py/nn/evaluate.py decides which targets use it.
    """

    def forward(self, y_pred, y_true, freqs, reduction='mean'):
        """
        Parameters
        ----------
        y_pred, y_true : torch.Tensor, shape (..., num_freqs)
            LOG-space (log-shape for target=='shape', log-spectral-energy
            for target=='density' — this project's convention throughout
            nn/training_loop.py/nn/evaluate.py) — exponentiated internally,
            since a CDF requires actual non-negative mass, not log-values.
            The internal per-spectrum mass-normalization (below) means this
            class is not actually shape-specific: it works identically for
            'density' target's physical log E(f), which is why it's used
            unchanged for both.
        freqs : torch.Tensor, shape (num_freqs,)
        reduction : 'mean' | 'none' | 'per_bin'
            'mean' (default): scalar, mean W1 distance over every (batch,
            lead_time, ...) axis — used as-is by the training loss.
            'none': returns the per-(batch, lead_time, ...) tensor before
            averaging — used by nn/evaluate.py's 'density' block, which
            must exclude near-zero-mass samples (M0_MASK_THRESHOLD) before
            averaging, the same masking already applied to Shape_RMSE/SS
            there; that requires per-sample values, not a pre-reduced scalar.
            'per_bin': returns |CDF_pred(f) - CDF_true(f)| itself, shape
            (..., num_freqs) — the pointwise CDF gap at each frequency
            BEFORE the final trapz integration collapses it to a scalar.
            W1 = trapz(this, freqs), so this is W1's per-bin breakdown: how
            much transport distance is attributable to each part of the
            spectrum, rather than how far off the raw value at that bin is
            (which is what an RMSE-per-bin measures). A peak that's shifted
            by one bin shows up here as a bump straddling the true peak's
            location, not a spike exactly at it — useful alongside an
            RMSE-per-bin plot precisely because RMSE punishes that shift as
            if the mass had vanished rather than moved.

        Returns
        -------
        torch.Tensor — scalar if reduction=='mean'; shape (batch,
        lead_time, ...) if reduction=='none'; shape (batch, lead_time, ...,
        num_freqs) if reduction=='per_bin'.
        """
        freqs = freqs.to(y_pred.device)
        pred_phys = torch.exp(y_pred)
        true_phys = torch.exp(y_true)

        pred_mass = torch.trapezoid(pred_phys, freqs, dim=-1).clamp(min=1e-8)
        true_mass = torch.trapezoid(true_phys, freqs, dim=-1).clamp(min=1e-8)
        pred_norm = pred_phys / pred_mass.unsqueeze(-1)
        true_norm = true_phys / true_mass.unsqueeze(-1)

        cdf_pred = _cumulative_trapz(pred_norm, freqs)
        cdf_true = _cumulative_trapz(true_norm, freqs)
        cdf_gap = torch.abs(cdf_pred - cdf_true)

        if reduction == 'per_bin':
            return cdf_gap

        w1 = torch.trapezoid(cdf_gap, freqs, dim=-1)
        if reduction == 'mean':
            return w1.mean()
        elif reduction == 'none':
            return w1
        else:
            raise ValueError(f"Unknown reduction {reduction!r}. Valid: 'mean', 'none', 'per_bin'")


def _trapz_bin_widths(freqs):
    """Per-bin trapezoidal half-width Δf_i for a frequency grid — the torch
    equivalent of utils.compute_hs.trapz_weights's numerator, computed on a
    freqs tensor supplied at call time (trapz_weights is numpy-only and
    expects a static precomputed array — the same reason
    SpectralWassersteinLoss above reimplements cumulative trapz rather than
    reusing a numpy helper).

    Deliberately NOT normalised to sum to 1 (unlike trapz_weights) — see
    SpectralKLDivergenceLoss's docstring: log_softmax's own normalising
    constant absorbs any uniform rescaling of Δf across bins identically,
    so normalising here would be a free no-op at the cost of an extra
    division per call.
    """
    delta_f = torch.empty_like(freqs)
    delta_f[0] = (freqs[1] - freqs[0]) / 2
    delta_f[-1] = (freqs[-1] - freqs[-2]) / 2
    delta_f[1:-1] = (freqs[2:] - freqs[:-2]) / 2
    return delta_f


class SpectralKLDivergenceLoss(torch.nn.Module):
    """
    Kullback-Leibler divergence D_KL(P_true ‖ P_pred) =
    Σ_i P_true(f_i) log(P_true(f_i) / P_pred(f_i)) between predicted and
    true spectra, treated as discrete probability distributions over
    frequency — a second alternative/complement to RMSELoss (see
    SpectralWassersteinLoss above for the first), aimed at the same
    multimodal-blur problem via a different mechanism again.

    P(f_i) is the Δf-weighted mass share of bin i:

        P(f_i) = phys(f_i) * Δf_i / Σ_j phys(f_j) * Δf_j

    (Δf via _trapz_bin_widths — same half-width trapezoidal scheme as
    utils.compute_hs.trapz_weights.) Δf-weighted rather than a flat 1/N
    share because these are DENSITIES on a non-uniform (log-spaced) grid: a
    bin's actual physical mass share also depends on how wide a frequency
    slice it represents — the same reasoning as SpectralWassersteinLoss's
    CDF and RMSELoss's frequency-weighted MSE.

    Because y_pred/y_true already ARE this project's native log-space
    representation (log-spectral-energy for target=='density', log-shape
    for target=='shape' — see utils.to_log_space), log P_pred/log P_true
    are obtained directly via:

        log P(f) = log_softmax(y + log(Δf), dim=-1)

    exactly the log-softmax + cross-entropy pattern from classification,
    applied over the frequency axis instead of over classes. Unlike
    SpectralWassersteinLoss's torch.exp(y_pred)/torch.exp(y_true), no
    explicit exp() of the raw log-space tensor is needed for either side
    except to recover P_true itself (KL's per-bin weight is P_true, not
    log P_true): log_softmax internally subtracts the per-row max before
    exponentiating, so it is unconditionally numerically stable regardless
    of y's value range — a nice-to-have property of this formulation, not
    a fix for an existing bug (SpectralWassersteinLoss's direct exp()
    hasn't been a problem in practice, since y_true is always floored —
    see utils.to_log_space).

    _trapz_bin_widths is NOT normalised to sum to 1 (unlike trapz_weights):
    log_softmax subtracts its own normalising constant regardless, and that
    constant absorbs any UNIFORM rescaling of Δf identically for every bin.
    Skipping the normalisation saves one division per call with no
    numerical downside (verified: normalised vs. unnormalised Δf give
    identical D_KL to 1e-5).

    Relationship to plain cross-entropy H(P_true, P_pred) =
    -Σ_i P_true(f_i) log P_pred(f_i): D_KL(P_true‖P_pred) =
    H(P_true, P_pred) - H(P_true), where H(P_true) is P_true's own Shannon
    entropy. Since H(P_true) does not depend on the prediction, KL and
    cross-entropy give IDENTICAL gradients w.r.t. y_pred — they differ only
    by a prediction-independent additive constant, so this is not a
    training-behavior choice. KL is implemented here (rather than raw
    cross-entropy) purely for the reported VALUE's interpretability:
    D_KL(P, P) = 0 exactly (Gibbs' inequality, equality iff P_pred ==
    P_true), matching SpectralWassersteinLoss's zero-for-identical-inputs
    convention — plain cross-entropy instead bottoms out at H(P_true), a
    nonzero, sample-dependent floor that makes "how close to perfect"
    harder to read off the raw loss value.

    Like a plain cross-entropy would be, this loss has NO spatial/ordering
    awareness: a coherently SHIFTED peak (all mass moved from bin i to a
    distant bin j) is punished just as catastrophically as under a plain
    pointwise loss — this does not rescue Wasserstein's target failure
    mode. Its distinct value is different: because the divergence's
    true-weighted sum is dominated by whatever probability P_pred assigns
    AT the bins where P_true is concentrated, it is comparatively
    insensitive to how the "wrong" residual probability is distributed
    among low-P_true bins — so a peak that BROADENS/partially collapses
    into its immediate neighbourhood (leaking predicted mass into several
    nearby bins while still leaving meaningful predicted mass exactly at
    the true peak) is discounted much more than under the current
    frequency-weighted log-space MSE, which separately penalises every one
    of those newly-elevated neighbouring bins. This does not fully "fix"
    local blur the way a dedicated curvature/smoothness term would — it
    only declines to pile on for it the way MSE does.
    """

    def forward(self, y_pred, y_true, freqs, reduction='mean'):
        """
        Parameters
        ----------
        y_pred, y_true : torch.Tensor, shape (..., num_freqs)
            LOG-space (see utils.to_log_space) — this project's native
            per-target representation; no exp() applied to either, per the
            log_softmax argument in the class docstring above.
        freqs : torch.Tensor, shape (num_freqs,)
        reduction : 'mean' | 'none' | 'per_bin'
            'mean' (default): scalar, mean KL divergence over every
            (batch, lead_time, ...) axis — used as-is by the training loss.
            'none': returns the per-(batch, lead_time, ...) tensor before
            averaging — mirrors SpectralWassersteinLoss's 'none', for any
            future masked-averaging caller (not currently wired to any
            such caller in nn/evaluate.py).
            'per_bin': returns P_true(f_i) * (log P_true(f_i) -
            log P_pred(f_i)) itself, shape (..., num_freqs) — the per-bin
            summand BEFORE the final sum over frequency.
            D_KL == per_bin.sum(-1).

        Returns
        -------
        torch.Tensor — scalar if reduction=='mean'; shape (batch,
        lead_time, ...) if reduction=='none'; shape (batch, lead_time, ...,
        num_freqs) if reduction=='per_bin'.
        """
        freqs = freqs.to(device=y_pred.device, dtype=y_pred.dtype)
        log_delta_f = torch.log(_trapz_bin_widths(freqs))

        log_p_pred = torch.log_softmax(y_pred + log_delta_f, dim=-1)
        log_p_true = torch.log_softmax(y_true + log_delta_f, dim=-1)
        p_true = torch.exp(log_p_true)

        # D_KL(P_true‖P_pred) per-bin summand: P_true(i) * (log P_true(i) -
        # log P_pred(i)). NOT negated — that would compute -D_KL (always
        # <= 0), which as an additive loss term would reward the optimizer
        # for making P_pred diverge further from P_true rather than
        # penalizing it.
        error = log_p_true - log_p_pred
        kl_per_bin = p_true * error

        if reduction == 'per_bin':
            return kl_per_bin

        kl = kl_per_bin.sum(dim=-1)
        if reduction == 'mean':
            return kl.mean()
        elif reduction == 'none':
            return kl
        else:
            raise ValueError(f"Unknown reduction {reduction!r}. Valid: 'mean', 'none', 'per_bin'")


class DirectionalLoss(torch.nn.Module):
    """Weighted composite loss over spectral density + directional wave
    parameters, per the meeting doc's Meta 3 proposal (2026-07-24 agenda):

        L = lambda_E * L_E + lambda_alpha1 * L_alpha1
                            + lambda_alpha2 * L_alpha2 + lambda_r * L_r

    Each L_x is a plain mean squared error (not RMSE, unlike RMSELoss above
    — matches the doc's literal formulas, which define every L_x as an
    unrooted MSE before the weighted sum). Angular error is represented via
    (sin, cos) channel MSE rather than atan2 circular distance: this avoids
    atan2's zero-gradient/discontinuity subtlety right at the wrap point,
    and matches the convention nn/channels.py already uses for the encoder's
    alpha_1/alpha_2 INPUT channels — see CHANNEL_SETS['full']. e.g. true
    alpha=1 deg vs pred alpha=359 deg (physically 2 deg apart) gives a small
    sin/cos MSE, not the ~128000 deg^2 a naive raw-degree MSE would produce.

    This class is not yet wired into any target/training path — see
    CLAUDE.md's Meta 3 discussion. It exists so the loss math itself can be
    validated (weighting behaviour, wraparound handling) ahead of a future
    'directional' target that would feed it real decoder output.
    """

    def __init__(self, lambda_E=1.0, lambda_alpha1=0.25, lambda_alpha2=0.25,
                 lambda_r=0.25):
        """
        Parameters
        ----------
        lambda_E, lambda_alpha1, lambda_alpha2, lambda_r : float, >= 0
            Term weights. Defaults treat density as the primary term
            (matching every existing target's loss, which is density-only)
            and the three directional terms as smaller regularisers rather
            than co-equal objectives — a starting point per deliverable 3.3,
            not a tuned value; adjust once a real 'directional' target
            exists to train against.
        """
        super().__init__()
        self.lambda_E = lambda_E
        self.lambda_alpha1 = lambda_alpha1
        self.lambda_alpha2 = lambda_alpha2
        self.lambda_r = lambda_r

    def forward(self, pred, true, freq_weights=None):
        """
        Parameters
        ----------
        pred, true : torch.Tensor, shape (..., num_freqs, 7)
            Last axis in CHANNEL_SETS['full'] order: [density, alpha_1_sin,
            alpha_1_cos, alpha_2_sin, alpha_2_cos, r_1, r_2].
        freq_weights : torch.Tensor | None, shape (num_freqs,)
            Optional trapezoidal frequency weights (e.g. utils.trapz_weights
            (freqs), sums to 1) applied identically to every term, matching
            the frequency-weighting convention the rest of the codebase uses
            for density/shape losses since the v7->v8 bump. When None,
            every bin is weighted equally — the doc's literal formula
            (flat 1/(N*tau*F) mean), used as the default here since this
            class isn't wired into training yet.

        Returns
        -------
        total : torch.Tensor, scalar — the weighted sum L.
        components : dict[str, torch.Tensor] — the four unweighted MSE
            terms (L_E, L_alpha1, L_alpha2, L_r), for logging/diagnostics.
        """
        def _mse(p, t):
            sq_err = (p - t) ** 2
            if freq_weights is None:
                return sq_err.mean()
            return (sq_err * freq_weights).sum(dim=-1).mean()

        L_E = _mse(pred[..., _DENSITY_IDX], true[..., _DENSITY_IDX])
        L_alpha1 = _mse(pred[..., _ALPHA1_SIN_IDX], true[..., _ALPHA1_SIN_IDX]) \
                 + _mse(pred[..., _ALPHA1_COS_IDX], true[..., _ALPHA1_COS_IDX])
        L_alpha2 = _mse(pred[..., _ALPHA2_SIN_IDX], true[..., _ALPHA2_SIN_IDX]) \
                 + _mse(pred[..., _ALPHA2_COS_IDX], true[..., _ALPHA2_COS_IDX])
        L_r = _mse(pred[..., _R1_IDX], true[..., _R1_IDX]) \
            + _mse(pred[..., _R2_IDX], true[..., _R2_IDX])

        total = (self.lambda_E * L_E
                 + self.lambda_alpha1 * L_alpha1
                 + self.lambda_alpha2 * L_alpha2
                 + self.lambda_r * L_r)

        components = {'L_E': L_E, 'L_alpha1': L_alpha1,
                      'L_alpha2': L_alpha2, 'L_r': L_r}
        return total, components


class SoftPeakHeightLoss(torch.nn.Module):
    """
    Differentiable "soft peak height" loss — a third auxiliary term,
    complementary to SpectralWassersteinLoss, aimed at a failure mode W1
    structurally underweights: a peak collapsing/broadening into its
    immediate neighbours while the total mass transported stays small
    (little distance travelled, even though the peak's own amplitude is
    destroyed). W1 measures how far mass moved; this measures whether the
    peak's own height survived, and is by construction invariant to pure
    translation (a shifted-but-otherwise-intact peak scores ~0 here,
    since it only compares a window's effective max against itself,
    positionless) — the two terms are meant to be summed, each covering
    the other's blind spot, not to replace one another.

    Motivation for softmax over max(): max() has (sub)gradient exactly 0
    everywhere except the single winning bin, and that winning bin can
    flip discontinuously between training steps — no usable gradient
    signal for a training loss. softmax turns the winner-take-all argmax
    into a normalised, confidence-weighted average of the spectrum
    against itself:

        sigma_k(i; tau) = exp(E(f_i)/tau) / sum_{j in window_k} exp(E(f_j)/tau)
        H_k(tau)        = sum_{i in window_k} E(f_i) * sigma_k(i; tau)

    tau interpolates between the two degenerate cases this replaces:
    tau -> infinity gives sigma_k -> uniform (H_k -> the window's plain
    mean); tau -> 0 gives sigma_k -> one-hot at the argmax (H_k -> max()).
    For any finite tau > 0 this is smooth — every bin in the window gets
    nonzero gradient, weighted by its own current softmax confidence (see
    tests below for the closed-form gradient identity this relies on:
    d H_k/d E_j = sigma_j * [1 + (E_j - H_k)/tau], which sums to exactly 1
    over the window).

    Per-peak temperature tau_k, corrected relative to a naive
    "tau_k = c * window_width_in_bins" formulation in two ways:
      1. Window width MUST be measured in Hz (freqs[right]-freqs[left]),
         not bin count — the buoy grid is log-spaced and non-uniform
         (dense ~0.005 Hz steps below 0.1 Hz, coarse ~0.02 Hz steps above
         0.365 Hz), so two partitions with the same bin count can span
         very different physical bandwidths depending on where they sit
         on the grid — the same caveat nn/freq_embedding.py's
         FreqDimEmbedding already documents for its own log-frequency
         encoding.
      2. tau must be scaled by the peak's own energy (H_k^true), not by
         window width alone — exp(E_i/tau) is only meaningful when tau is
         in E's own units (spectral density, e.g. m^2/Hz), not frequency's.
         A width-only tau makes the softmax's sharpness swing wildly
         between calm and stormy samples that happen to have similarly
         SHAPED (same relative bandwidth) partitions, since E's absolute
         scale has nothing to do with a window's width in Hz — e.g. the
         same window width would leave a calm-sea peak's softmax sensibly
         soft while making a storm sample's near-hard-argmax again (the
         exact problem softmax was introduced to avoid), even though nothing
         about the PARTITION's shape changed, only the sea state's energy.
    Combining both, tau_k is a fixed fraction of the peak's own height,
    modulated by that peak's bandwidth relative to the grid's total span
    (dimensionless ratio) — narrow (swell) partitions get a small,
    sharp/near-max tau; wide (wind-sea) partitions get a larger, softer
    tau, matching the fact that a broad wind-sea peak genuinely doesn't
    have as sharply-defined a "height" as a narrow swell peak:

        tau_k = max(tau_min, c * H_k^true * (freqs[r_k]-freqs[l_k]) / freq_ref)

    H_k^true only ever appears on the loss side, never inside
    model.infer() — using it to calibrate tau is not a train/inference
    leak, the same category of "read the label" already done by the
    (also label-only) window detection itself
    (utils.spectral_partitioning.find_peak_windows).

    Known asymmetry: H_k^true is a HARD max (no gradient needed on the
    label side) while H_k^pred is a SOFT, temperature-weighted estimate of
    the same kind of values — and a soft estimate is generically <= the
    hard max of the values it's drawn from. Consequently this loss's
    minimum is not exactly at y_pred == y_true: even a PERFECT prediction
    leaves a small positive residual (shrinking as tau_k shrinks) — see
    tests/test_loss.py::TestSoftPeakHeightLoss::
    test_perfect_prediction_residual_is_small_but_nonzero. This is an
    intentional trade-off, not a bug (the alternative — softening the
    label side too, so both sides use an identical formula — would remove
    the residual but reintroduces a design question the source discussion
    explicitly avoided: the label doesn't need gradient, so there's no
    reason to pay for softening it). In practice the residual stays small
    relative to a real collapse-type error (same test file,
    test_local_collapse_scores_worse_than_shift_unlike_wasserstein) — but
    if this is ever wired into training, log it as its own component
    (mirroring DirectionalLoss's `components` return) so a training run
    isn't misread as having a nonzero floor it can never cross.

    Windows (left_idx/right_idx per peak) are NOT computed here. Peak
    detection (utils.spectral_partitioning.find_significant_peaks /
    find_peak_windows) is scipy-based, single-spectrum, non-differentiable
    Python looping — nn/evaluate.py:53-60 documents why this project
    already treats it as opt-in/evaluation-only rather than something run
    every training step. This class expects left_idx/right_idx/peak_mask
    already computed (once per sample — mirroring how freq_means/
    shape_means are computed once in
    nn/optimization.py::_prepare_dataloaders and threaded through
    training rather than recomputed per batch — via find_peak_windows,
    padded to a fixed max_peaks and batched into tensors) and never
    derives them internally.
    """

    def __init__(self, c=0.15, tau_min=1e-4, freq_ref=None):
        """
        Parameters
        ----------
        c : float, default 0.15
            Fraction of a peak's own true height used as its softmax
            temperature scale, before the width modulation — a starting
            hyperparameter, not yet tuned; adjust by validation once this
            is wired into training (see the class docstring's tau_k
            derivation).
        tau_min : float, default 1e-4
            Absolute numerical floor on tau_k (same spirit as
            utils.log_transform.LOG_FLOOR_FRACTION's floor), preventing a
            near-zero-energy or degenerate-width peak from collapsing tau
            toward zero, which would push the softmax back toward the
            zero-gradient hard max it exists to avoid.
        freq_ref : float | None, default None
            Reference bandwidth [Hz] each peak's window width is divided
            by to get a dimensionless width ratio. None (default) uses the
            full grid span (freqs[-1]-freqs[0]) at forward() call time —
            the natural zero-config choice, since a peak's window can
            never be wider than the whole spectrum.
        """
        super().__init__()
        self.c = c
        self.tau_min = tau_min
        self.freq_ref = freq_ref

    def forward(self, y_pred, y_true, freqs, left_idx, right_idx, peak_mask,
                reduction='mean'):
        """
        Parameters
        ----------
        y_pred, y_true : torch.Tensor, shape (..., num_freqs)
            LOG-space (see utils.to_log_space), same convention as
            SpectralWassersteinLoss/SpectralKLDivergenceLoss — exponentiated
            internally to get physical E(f), since a peak "height" is only
            meaningful in physical (not log) space.
        freqs : torch.Tensor, shape (num_freqs,)
        left_idx, right_idx : torch.LongTensor, shape (..., max_peaks)
            INCLUSIVE bin-index bounds of each peak's trough-to-trough
            window (utils.spectral_partitioning.find_peak_windows),
            broadcastable against y_pred/y_true's leading (batch,
            lead_time, ...) dims plus a trailing max_peaks axis. Padding
            slots (samples with fewer than max_peaks real peaks) may use
            any placeholder, in-range or not (e.g. 0, 0 or -1, -1) — both
            indices are clamped into range internally, and right_idx is
            floored at left_idx, so every window covers at least one bin
            and no downstream softmax row is ever entirely masked out.
        peak_mask : torch.BoolTensor, shape (..., max_peaks)
            True at slots holding a real detected peak; False at padding
            slots, excluded from the aggregation below (not zero-filled —
            see the K=0 handling under 'mean').
        reduction : 'mean' | 'none' | 'per_peak'
            'mean' (default): scalar — for each sample, the mean squared
            height error over that sample's real peaks (K=0 samples, i.e.
            peak_mask all False, are EXCLUDED rather than contributing a
            filled-in zero — mirrors nn/evaluate.py's M0_MASK_THRESHOLD/
            valid.any() pattern for a quantity undefined for that sample,
            rather than diluting the mean toward "no error" for a sample
            this loss has nothing to say about); NaN only if literally no
            sample in the batch has any peak.
            'none': shape (...,) — the per-sample mean-over-real-peaks
            value (same masking as 'mean', before the final batch mean) —
            mirrors the sibling losses' 'none', for a future masked-
            averaging caller (e.g. an nn/evaluate.py block).
            'per_peak': shape (..., max_peaks) — the raw (unmasked)
            per-peak squared height error, before aggregating over peaks —
            mirrors SpectralWassersteinLoss's 'per_bin': the breakdown
            BEFORE the final reduction: caller applies peak_mask itself.

        Returns
        -------
        torch.Tensor — scalar if reduction=='mean'; shape (...,) if
        'none'; shape (..., max_peaks) if 'per_peak'.
        """
        freqs = freqs.to(device=y_pred.device, dtype=y_pred.dtype)
        num_freqs = y_pred.shape[-1]

        # Defensive clamp: guarantees left_idx <= right_idx and both in
        # range, so every (..., peak) window covers at least one bin — no
        # downstream op ever sees an all-masked-out softmax row, whatever
        # placeholder padding slots use upstream (see left_idx/right_idx
        # docstring above).
        left_idx = left_idx.clamp(0, num_freqs - 1)
        right_idx = right_idx.clamp(0, num_freqs - 1)
        right_idx = torch.maximum(right_idx, left_idx)

        bin_pos = torch.arange(num_freqs, device=y_pred.device)
        in_window = (bin_pos >= left_idx.unsqueeze(-1)) & (bin_pos <= right_idx.unsqueeze(-1))
        # (..., max_peaks, num_freqs)

        E_pred = torch.exp(y_pred).unsqueeze(-2).expand_as(in_window)  # (..., max_peaks, num_freqs)
        E_true = torch.exp(y_true).unsqueeze(-2).expand_as(in_window)

        # finfo.min rather than literal -inf: softmax subtracts the row
        # max before exponentiating, so masked entries still underflow
        # cleanly to a weight of exactly 0.0 — but a literal -inf risks a
        # 0 * inf -> NaN downstream (e.g. via peak_mask) that finfo.min,
        # being finite, cannot produce.
        neg_inf = torch.finfo(y_pred.dtype).min
        H_true = torch.where(in_window, E_true, neg_inf).amax(dim=-1)  # (..., max_peaks)
        # Label side: hard max, no gradient needed (see class docstring).

        delta_f_k = freqs[right_idx] - freqs[left_idx]  # (..., max_peaks), Hz
        freq_ref = self.freq_ref if self.freq_ref is not None else (freqs[-1] - freqs[0])
        tau_k = (self.c * H_true * (delta_f_k / freq_ref)).clamp(min=self.tau_min)

        logits = torch.where(in_window, E_pred / tau_k.unsqueeze(-1), neg_inf)
        weights = torch.softmax(logits, dim=-1)
        H_pred = (E_pred * weights).sum(dim=-1)  # (..., max_peaks)

        per_peak_sq_err = (H_pred - H_true) ** 2

        if reduction == 'per_peak':
            return per_peak_sq_err

        peak_mask_f = peak_mask.to(per_peak_sq_err.dtype)
        n_peaks = peak_mask_f.sum(dim=-1)  # (...,)
        per_sample = (per_peak_sq_err * peak_mask_f).sum(dim=-1) / n_peaks.clamp(min=1)
        has_peak = n_peaks > 0

        if reduction == 'none':
            return per_sample
        elif reduction == 'mean':
            if not bool(has_peak.any()):
                return per_sample.new_tensor(float('nan'))
            return per_sample[has_peak].mean()
        else:
            raise ValueError(f"Unknown reduction {reduction!r}. Valid: 'mean', 'none', 'per_peak'")