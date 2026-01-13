from pathlib import Path
from typing import Union

import torch

from models.three_d.original_3dgs.model import GaussianModel


@torch.no_grad()
def load_3dgs_model_original(
    model_dir: Union[str, Path],
    device: Union[str, torch.device] = "cuda",
    iteration: int = 30_000,
) -> GaussianModel:
    """Load a Gaussian Splatting model produced by the original 3DGS repository."""

    model_path = Path(model_dir)
    assert model_path.is_dir(), f"3DGS model directory does not exist: {model_dir}"

    assert iteration >= 0, "Iteration must be non-negative"
    ply_path = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    assert ply_path.is_file(), (
        "3DGS model directory does not contain expected point cloud file: "
        f"{ply_path}"
    )

    device = torch.device(device)

    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(str(ply_path), device=device)

    return gaussians.to(device)
