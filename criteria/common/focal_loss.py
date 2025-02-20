from typing import Optional
import torch
import torch.nn.functional as F
from criteria.wrappers import SingleTaskCriterion
from utils.input_checks import check_classification, check_semantic_segmentation


class FocalLoss(SingleTaskCriterion):

    def __init__(self, alpha: Optional[float] = 1.0, beta: Optional[float] = 2.0):
        """
        Initialize the FocalLoss with alpha and beta parameters.
        Args:
            alpha (float | int): Balancing factor for class imbalance. Default is 1.0.
            beta (float | int): Focusing parameter to adjust the rate at which
                          easy examples are down-weighted. Default is 2.0.
        """
        super(FocalLoss, self).__init__()
        assert isinstance(alpha, (float, int)), f"{type(alpha)=}"
        self.alpha = alpha
        assert isinstance(beta, (float, int)), f"{type(beta)=}"
        self.beta = beta

    @staticmethod
    def _prepare_focal(tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.ndim in {3, 4}
        if tensor.ndim == 3:
            tensor = tensor.flatten()
        elif tensor.ndim == 4:
            N, C, H, W = tensor.shape
            tensor = tensor.view((N, C, H*W))
            tensor = tensor.transpose(1, 2)
            tensor = tensor.contiguous().view(N*H*W, C)
        return tensor

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
        try:
            check_classification(y_pred, y_true)
        except:
            check_semantic_segmentation(y_pred, y_true)
            y_pred = self._prepare_focal(y_pred)
            y_true = self._prepare_focal(y_true)
            check_classification(y_pred, y_true)
        assert y_pred.size(1) == 2  # support only binary case

        # Convert logits to probabilities using softmax
        probs = F.softmax(y_pred, dim=-1)

        # Select the probabilities of the correct classes
        true_probs = probs[torch.arange(y_true.size(0)), y_true]

        # Compute the focal loss
        focal_weight = self.alpha * (1 - true_probs) ** self.beta
        loss = -focal_weight * torch.log(true_probs + 1e-12)  # Avoid log(0)

        # Return the mean loss over the batch
        return loss.mean()
