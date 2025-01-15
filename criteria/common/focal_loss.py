from typing import Optional
import torch
import torch.nn.functional as F
from criteria.wrappers import SingleTaskCriterion


class FocalLoss(SingleTaskCriterion):

    def __init__(self, alpha: Optional[float] = 1.0, beta: Optional[float] = 2.0):
        """
        Initialize the FocalLoss with alpha and beta parameters.
        Args:
            alpha (float): Balancing factor for class imbalance. Default is 1.0.
            beta (float): Focusing parameter to adjust the rate at which
                          easy examples are down-weighted. Default is 2.0.
        """
        super(FocalLoss, self).__init__()
        assert isinstance(alpha, float), f"{type(alpha)=}"
        self.alpha = alpha
        assert isinstance(beta, float), f"{type(beta)=}"
        self.beta = beta

    def _compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute the focal loss.
        Args:
            y_pred (torch.Tensor): Predicted logits (shape: [batch_size, num_classes]).
            y_true (torch.Tensor): Ground truth labels (shape: [batch_size]).
        Returns:
            torch.Tensor: Scalar focal loss value.
        """
        # Input checks
        assert isinstance(y_pred, torch.Tensor) and y_pred.ndim == 2
        assert isinstance(y_true, torch.Tensor) and y_true.ndim == 1
        assert y_pred.size(0) == y_true.size(0), f"{y_pred.shape=}, {y_true.shape=}"

        # Convert logits to probabilities using softmax
        probs = F.softmax(y_pred, dim=-1)

        # Select the probabilities of the correct classes
        true_probs = probs[torch.arange(y_true.size(0)), y_true]

        # Compute the focal loss
        focal_weight = self.alpha * (1 - true_probs) ** self.beta
        loss = -focal_weight * torch.log(true_probs + 1e-12)  # Avoid log(0)

        # Return the mean loss over the batch
        return loss.mean()
