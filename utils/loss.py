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

    def forward(self, y_pred, y_true, weights=None):
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
        """
        sq_err = (y_pred - y_true) ** 2
        if weights is None:
            mse = sq_err.mean()
        else:
            mse = (sq_err * weights).sum(dim=-1).mean()
        return torch.sqrt(mse)


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