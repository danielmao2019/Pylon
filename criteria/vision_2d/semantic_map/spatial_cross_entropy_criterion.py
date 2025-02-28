from typing import Tuple, Optional
import torch
import torchvision
from criteria.wrappers.single_task_criterion import SingleTaskCriterion
from utils.input_checks import check_semantic_segmentation


class SpatialCrossEntropyCriterion(SingleTaskCriterion):

    def __init__(
        self,
        ignore_index: int,
        class_weights: Optional[Tuple[float, ...]] = None,
        device: Optional[torch.device] = torch.device('cuda'),
    ) -> None:
        super(SpatialCrossEntropyCriterion, self).__init__()
        self.ignore_index = ignore_index
        if class_weights is not None:
            assert type(class_weights) == tuple, f"{type(class_weights)=}"
            assert all([type(elem) == float for elem in class_weights])
            class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        self.class_weights = class_weights

    def _compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            y_pred (torch.Tensor): a float32 tensor of shape (N, C, H, W) for predicted logits.
            y_true (torch.Tensor): an int64 tensor of shape (N, H, W) for ground-truth mask.

        Returns:
            loss (torch.Tensor): a float32 scalar tensor for loss value.
        """
        # Input checks
        check_semantic_segmentation(y_pred=y_pred, y_true=y_true)

        # Match resolution
        if y_pred.shape[-2:] != y_true.shape[-2:]:
            y_pred = torchvision.transforms.Resize(
                size=y_true.shape[-2:], interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST,
            )(y_pred)

        # Define criterion
        class_weights = self.class_weights
        if class_weights is None:
            num_classes = y_pred.shape[-3]
            with torch.no_grad():
                valid_mask = (y_true != self.ignore_index)
                class_counts = torch.bincount(y_true[valid_mask].view(-1), minlength=num_classes).float()
                total_pixels = valid_mask.sum()
                class_weights = (total_pixels - class_counts) / total_pixels
                class_weights[class_counts == 0] = 0  # Avoid NaN for missing classes
        criterion = torch.nn.CrossEntropyLoss(
            weight=class_weights, ignore_index=self.ignore_index, reduction='mean',
        )

        # Compute loss
        return criterion(input=y_pred, target=y_true)
