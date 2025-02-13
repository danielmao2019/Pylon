import torch
from utils.input_checks import check_semantic_segmentation


class IoULoss(torch.nn.Module):
    """
    Intersection over Union (IoU) Loss.
    """
    def __init__(self, reduction: str = "mean") -> None:
        super(IoULoss, self).__init__()
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError("Invalid reduction type. Choose from 'mean', 'sum', or 'none'.")
        self.reduction = reduction

    def _compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Computes IoU loss given prediction and target tensors.
        """
        # input checks
        check_semantic_segmentation(y_pred=y_pred, y_true=y_true)

        B, C, _, _ = y_pred.shape
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)
        assert torch.all(torch.isclose(torch.sum(y_pred, dim=1, keepdim=True), torch.ones_like(y_pred)))
        y_true = torch.eye(C, device=y_true.device)[y_true]
        y_true = y_true.permute(0, 3, 1, 2)

        intersection = (y_true * y_pred).sum(dim=[2, 3])
        union = y_true.sum(dim=[2, 3]) + y_pred.sum(dim=[2, 3]) - intersection
        iou = intersection / union
        assert iou.shape == (B, C), f"{iou.shape=}"

        iou_loss = 1 - iou.mean(dim=1)

        if self.reduction == "mean":
            return iou_loss.mean()
        elif self.reduction == "sum":
            return iou_loss.sum()
        else:
            return iou_loss
