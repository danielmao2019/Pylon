from typing import Tuple, Optional
import torch
import torchvision
from criteria.wrappers.single_task_criterion import SingleTaskCriterion
from utils.input_checks import check_semantic_segmentation


class SemanticSegmentationCriterion(SingleTaskCriterion):

    def __init__(
        self,
        ignore_index: Optional[int] = None,
        class_weights: Optional[Tuple[float, ...]] = None,
        device: Optional[torch.device] = torch.device('cuda'),
    ) -> None:
        super(SemanticSegmentationCriterion, self).__init__()
        if ignore_index is None:
            ignore_index = 255
        if class_weights is not None:
            assert type(class_weights) == tuple, f"{type(class_weights)=}"
            assert all([type(elem) == float for elem in class_weights])
            class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index, weight=class_weights, reduction='mean',
        )

    def _compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            y_pred (torch.Tensor): a float32 tensor of shape (N, C, H, W) for predicted logits.
            y_true (torch.Tensor): an int64 tensor of shape (N, H, W) for ground-truth mask.

        Returns:
            loss (torch.Tensor): a float32 scalar tensor for loss value.
        """
        # input checks
        check_semantic_segmentation(y_pred=y_pred, y_true=y_true)
        # match resolution
        if y_pred.shape[-2:] != y_true.shape[-2:]:
            y_pred = torchvision.transforms.Resize(
                size=y_true.shape[-2:], interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST,
            )(y_pred)
        # compute loss
        return self.criterion(input=y_pred, target=y_true)
