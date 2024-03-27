import torch


class NormalEstimationLoss:

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            y_pred (torch.Tensor): the (unnormalized) output of the model.
            y_true (torch.Tensor): the ground truth label for the normal vector.
        Returns:
            loss (torch.Tensor): a single-element tensor representing the loss for normal estimation.
        """
        # convert model outputs to predictions
        y_pred = y_pred / torch.norm(y_pred, p=2, dim=1, keepdim=True)
        # sanity checks
        assert y_pred.shape == y_true.shape, f"{y_pred.shape=}, {y_true.shape=}"
        assert len(y_pred.shape) == len(y_true.shape) == 4, f"{y_pred.shape=}, {y_true.shape=}"
        assert torch.all((torch.norm(y_pred, p=2, dim=1) - 1).abs() < 1e-05), f"{(torch.norm(y_pred, p=2, dim=1) - 1).abs().max()=}"
        assert torch.all((torch.norm(y_true, p=2, dim=1) - 1).abs() < 1e-05), f"{(torch.norm(y_true, p=2, dim=1) - 1).abs().max()=}"
        # compute loss
        binary_mask = (torch.sum(y_true, dim=1) != 0).float().unsqueeze(1)
        loss = 1 - torch.sum((y_pred * y_true) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)
        assert loss.numel() == 1, f"{loss.shape=}"
        return loss
