from pathlib import Path
from typing import Union

import torch

from models.three_d.cheng2025clod.model import Cheng2025CLODGS


@torch.no_grad()
def load_cheng2025_clod_model(
    model_dir: Union[str, Path],
    device: Union[str, torch.device] = "cuda",
    iteration: int = 30_000,
    sh_degree: int = 3,
) -> Cheng2025CLODGS:
    """Load a Cheng2025CLOD checkpoint from disk."""
    # Input validations
    model_path = Path(model_dir)
    assert (
        model_path.is_dir()
    ), f"Cheng2025CLOD model directory does not exist: {model_dir}"
    assert isinstance(iteration, int), f"{type(iteration)=}"
    assert iteration >= 0, f"{iteration=}"
    ply_path = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    assert ply_path.is_file(), f"point_cloud.ply missing at {ply_path}"

    device = torch.device(device)

    gaussians = Cheng2025CLODGS(sh_degree=sh_degree)
    gaussians.load_ply(str(ply_path), device=device)

    return gaussians.to(device)
