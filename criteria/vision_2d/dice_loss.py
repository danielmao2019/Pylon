from typing import Optional
import torch

class DiceLoss(torch.nn.Module):
    """Dice loss, need one hot encode input
    Args:
        eps: added to the denominator for numerical stability.
    Reference:
        * https://github.com/likyoo/Siam-NestedUNet/blob/master/utils/metrics.py
    """
    def __init__(self, eps: Optional[float] = 1e-7):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Computes the Sørensen–Dice loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the dice loss so we
        return the negated dice loss.
        Args:
            y_true: a tensor of shape [B, H, W].
            y_pred: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or y_pred of the model.
        Returns:
            dice_loss: the Sørensen–Dice loss.
        """
        assert isinstance(y_pred, torch.Tensor), "y_pred must be a PyTorch tensor."
        assert isinstance(y_true, torch.Tensor), "y_true must be a PyTorch tensor."
        assert y_pred.ndimension() == 4, "y_pred must have shape [B, C, H, W]."
        assert y_true.ndimension() == 3, "y_true must have shape [B, H, W]."
        #assert y_pred.shape == y_true.shape
        #onehot encoding
        num_classes = y_pred.shape[1]

        #use onehot in scikit

        true_1_hot = torch.eye(num_classes)[y_true]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        
        probas = torch.nn.functional.softmax(y_pred, dim=1)
        assert all(torch.isclose(torch.sum(probas, dim=1) == 1, torch.ones(size=(y_pred.shape[2], y_pred.shape[3]), dtype=torch.bool)).tolist())

        dims = (y_pred.shape[0],) + tuple(range(2, y_true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality)).mean()
        return (1 - dice_loss)
