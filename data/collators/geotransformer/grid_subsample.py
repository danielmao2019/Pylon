import importlib
import sys
from pathlib import Path
from typing import Tuple

import torch


def grid_subsample(
    points: torch.Tensor, lengths: torch.Tensor, voxel_size: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Grid subsampling in stack mode.

    This function is implemented on CPU.

    Args:
        points (Tensor): stacked points. (N, 3)
        lengths (Tensor): number of points in the stacked batch. (B,)
        voxel_size (float): voxel size.

    Returns:
        s_points (Tensor): stacked subsampled points (M, 3)
        s_lengths (Tensor): numbers of subsampled points in the batch. (B,)
    """
    geotransformer_path = (
        Path(__file__).resolve().parents[3] / 'data' / 'collators' / 'geotransformer'
    )
    if str(geotransformer_path) not in sys.path:
        sys.path.append(str(geotransformer_path))
    ext_module = importlib.import_module('geotransformer.ext')

    device = points.device
    s_points, s_lengths = ext_module.grid_subsampling(
        points.cpu(), lengths.cpu(), voxel_size
    )
    return s_points.to(device), s_lengths.to(device)
