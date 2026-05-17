"""Shared class-color palette helpers for atomic displays."""

from typing import Dict, Tuple

import torch


def get_class_color(class_id: int) -> Tuple[int, int, int]:
    """Map a class identifier to a stable RGB color.

    Args:
        class_id: Integer class identifier.

    Returns:
        RGB color tuple with channel values in `[0, 255]`.
    """
    assert isinstance(class_id, int), "Class id must be an integer. class_id=%r" % (
        class_id,
    )
    assert class_id >= 0, "Class id must be non-negative. class_id=%r" % class_id

    palette = [
        (37, 99, 235),
        (220, 38, 38),
        (22, 163, 74),
        (202, 138, 4),
        (147, 51, 234),
        (8, 145, 178),
        (234, 88, 12),
        (79, 70, 229),
    ]
    return palette[class_id % len(palette)]


def map_class_ids_to_rgb(class_ids: torch.Tensor) -> Dict[int, Tuple[int, int, int]]:
    """Map class identifiers to stable RGB colors.

    Args:
        class_ids: Tensor containing integer class identifiers.

    Returns:
        Mapping from class identifier to RGB color tuple with channel values in `[0, 255]`.
    """
    assert isinstance(class_ids, torch.Tensor), (
        "Class ids must be a torch.Tensor. class_ids=%r" % class_ids
    )
    assert class_ids.numel() > 0, (
        "Class ids tensor must be non-empty. class_ids=%r" % class_ids
    )

    flattened_class_ids = class_ids.detach().cpu().reshape(-1).to(torch.int64)
    unique_class_ids = torch.unique(flattened_class_ids, sorted=True)
    return {
        int(class_id.item()): get_class_color(class_id=int(class_id.item()))
        for class_id in unique_class_ids
    }
