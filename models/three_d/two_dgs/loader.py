"""Utilities for loading 2D Gaussian Splatting checkpoints without repo deps."""

from pathlib import Path

import torch
from models.three_d.two_dgs.model import GaussianModel


@torch.no_grad()
def load_2dgs_model(
    model_dir: str | Path,
    device: str | torch.device = "cuda",
) -> GaussianModel:
    model_path = Path(model_dir)
    if not model_path.is_dir():
        raise FileNotFoundError(f"2DGS model directory does not exist: {model_dir}")

    ply_relpath = Path("point_cloud") / "iteration_30000" / "point_cloud.ply"
    ply_path = model_path / ply_relpath
    if not ply_path.is_file():
        raise FileNotFoundError(
            "2DGS model directory does not contain expected point cloud file: "
            f"{ply_path}"
        )

    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(str(ply_path))

    gaussian_device = gaussians.get_xyz.device
    target_device = torch.device(device)
    if target_device.type != gaussian_device.type:
        raise ValueError(
            "Loaded 2DGS model resides on device '{}' but caller requested '{}'. "
            "2DGS checkpoints are stored on GPU only.".format(
                gaussian_device, target_device
            )
        )
    return gaussians
