"""Implementation largely based on https://github.com/facebookresearch/detectron2/blob/main/detectron2/structures/boxes.py.
"""
import torch


def pairwise_intersection(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the intersection area between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax)

    Args:
        boxes1, boxes2 (torch.Tensor): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: intersection, sized [N,M].
    """
    # input checks
    assert type(boxes1) == torch.Tensor, f"{type(boxes1)=}"
    assert boxes1.dim() == 2 and boxes1.shape[1] == 4, f"{boxes1.shape=}"
    assert type(boxes2) == torch.Tensor, f"{type(boxes2)=}"
    assert boxes2.dim() == 2 and boxes2.shape[1] == 4, f"{boxes2.shape=}"
    # compute intersection
    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    width_height.clamp_(min=0)  # [N,M,2]
    intersection = width_height.prod(dim=2)  # [N,M]
    assert intersection.shape == (len(boxes1), len(boxes1)), f"{intersection.shape=}, {boxes1.shape=}, {boxes2.shape=}"
    assert torch.all(intersection >= 0), f"{intersection.min()=}, {intersection.max()=}"
    return intersection


def get_areas(boxes: torch.Tensor) -> torch.Tensor:
    """
    Computes the area of all the boxes.

    Returns:
        torch.Tensor: a vector with areas of each box.
    """
    assert type(boxes) == torch.Tensor, f"{type(boxes)=}"
    assert boxes.dim() == 2 and boxes.shape[1] == 4, f"{boxes.shape=}"
    heights = boxes[:, 3] - boxes[:, 1]
    widths = boxes[:, 2] - boxes[:, 0]
    areas = heights * widths
    assert areas.shape == (len(boxes),), f"{areas.shape=}, {boxes.shape=}"
    assert torch.all(areas >= 0), f"{areas.min()=}, {areas.max()=}"
    return areas


def pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M, compute the IoU
    (intersection over union) between **all** N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1, boxes2 (torch.Tensor): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    # input checks
    assert type(boxes1) == torch.Tensor, f"{type(boxes1)=}"
    assert boxes1.dim() == 2 and boxes1.shape[1] == 4, f"{boxes1.shape=}"
    assert type(boxes2) == torch.Tensor, f"{type(boxes2)=}"
    assert boxes2.dim() == 2 and boxes2.shape[1] == 4, f"{boxes2.shape=}"
    # compute IoU
    areas1 = get_areas(boxes1)  # [N]
    areas2 = get_areas(boxes2)  # [M]
    inter = pairwise_intersection(boxes1, boxes2)
    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (areas1[:, None] + areas2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou
