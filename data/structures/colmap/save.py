from pathlib import Path
from typing import Any

from data.structures.colmap.colmap import COLMAP_Data
from data.structures.colmap.load import save_model_binary, save_model_text


def save_colmap_binary(colmap: Any, output_dir: str | Path) -> None:
    # Input validations
    assert isinstance(colmap, COLMAP_Data), f"{type(colmap)=}"
    assert isinstance(output_dir, (str, Path)), f"{type(output_dir)=}"

    save_model_binary(
        output_dir=output_dir,
        cameras=colmap.cameras,
        images=colmap.images,
        points3D=colmap.points3D,
    )


def save_colmap_text(colmap: Any, output_dir: str | Path) -> None:
    # Input validations
    assert isinstance(colmap, COLMAP_Data), f"{type(colmap)=}"
    assert isinstance(output_dir, (str, Path)), f"{type(output_dir)=}"

    save_model_text(
        output_dir=output_dir,
        cameras=colmap.cameras,
        images=colmap.images,
        points3D=colmap.points3D,
    )
