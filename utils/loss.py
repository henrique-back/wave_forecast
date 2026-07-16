import torch

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