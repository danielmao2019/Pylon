from typing import Tuple, Optional
import torch
import torchvision
from criteria.wrappers import SingleTaskCriterion
from utils.input_checks import check_semantic_segmentation


class DiceLoss(SingleTaskCriterion):
    __doc__ = """
    Reference:
        * https://github.com/likyoo/Siam-NestedUNet/blob/master/utils/metrics.py
    """

    REDUCTION_OPTIONS = ['mean', 'sum', 'none']

    def __init__(
        self,
        class_weights: Optional[Tuple[float, ...]] = None,
        reduction: Optional[str] = 'mean',
        device: Optional[torch.device] = torch.device('cuda'),
    ) -> None:
        super(DiceLoss, self).__init__()
        if class_weights is not None:
            assert type(class_weights) == tuple, f"{type(class_weights)=}"
            assert all([type(elem) == float for elem in class_weights])
            class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        self.class_weights = class_weights
        assert reduction in self.REDUCTION_OPTIONS, f"{reduction=}"
        self.reduction = reduction

    def _compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        r"""Computes the Sørensen-Dice loss.
        Args:
            y_true: a tensor of shape [B, H, W].
            y_pred: a tensor of shape [B, C, H, W]. Corresponds to output logits from the model.
        Returns:
            dice_loss: the Sørensen-Dice loss.
        """
        # input checks
        check_semantic_segmentation(y_pred=y_pred, y_true=y_true)
        # match resolution
        if y_pred.shape[-2:] != y_true.shape[-2:]:
            y_pred = torchvision.transforms.Resize(
                size=y_true.shape[-2:], interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST,
            )(y_pred)
        # initialization
        B, C, _, _ = y_pred.shape

        # one-hot encoding
        y_true = torch.eye(C, dtype=torch.float32, device=y_true.device)[y_true]
        y_true = y_true.permute(0, 3, 1, 2)

        # convert logits to probability distribution
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)
        assert torch.all(torch.isclose(torch.sum(y_pred, dim=1, keepdim=True), torch.ones_like(y_pred)))

        # compute Dice Loss
        inter = torch.sum(y_pred * y_true, dim=(2, 3))
        total = torch.sum(y_pred + y_true, dim=(2, 3))
        class_dice_loss = 1 - 2 * inter / total
        assert class_dice_loss.shape == (B, C), f"{class_dice_loss.shape=}"
        dice_loss = torch.sum(self.class_weights.view((B, C)) * class_dice_loss, dim=1)
        assert dice_loss.shape == (B,), f"{dice_loss.shape=}"
        if self.reduction == 'mean':
            dice_loss = dice_loss.mean(dim=0)
        elif self.reduction == 'sum':
            dice_loss = dice_loss.sum(dim=0)
        else:
            pass
        assert dice_loss.shape == (() if self.reduction != 'none' else (B,)), f"{dice_loss.shape=}, {self.reduction=}"
        return dice_loss
