"""Example point cloud rendering pipeline for iVISION-PCR."""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.point_cloud import load_point_cloud
from data.structures.three_d.point_cloud.ops.rendering.render_rgb import (
    render_rgb_from_point_cloud,
)
from data.structures.three_d.point_cloud.ops.rendering.render_segmentation import (
    render_segmentation_from_point_cloud,
)


def save_rendered_image(image_tensor: torch.Tensor, filepath: Path) -> None:
    """Save a rendered RGB tensor ([3, H, W], values in [0, 1]) as a PNG."""
    image = image_tensor.detach().clamp(0.0, 1.0).cpu()
    array = (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    Image.fromarray(array, mode="RGB").save(filepath)


def save_mask(mask_tensor: torch.Tensor, filepath: Path) -> None:
    """Save a rendered mask/segmentation tensor ([H, W], bool or int) as a PNG."""
    mask = mask_tensor.detach().cpu()
    if mask.dtype == torch.bool:
        array = (mask.numpy().astype(np.uint8) * 255)
        Image.fromarray(array, mode="L").save(filepath)
    else:
        array = mask.to(torch.int64).numpy().astype(np.int32)
        Image.fromarray(array, mode="I").save(filepath)


def demo_rendering() -> None:
    """Render RGB and segmentation images from a dummy point cloud setup."""
    device = torch.device("cuda")
    point_cloud_path = Path("/path/to/dummy_point_cloud.ply")
    # Point clouds must follow supported formats (.ply/.las/.laz/.off/.txt/.pth) with 'rgb' and
    # segmentation labels so downstream renderers can consume them.

    # Camera intrinsics must be 3x3 pinhole matrices; keep them on the GPU for rendering.
    fx, fy, cx, cy = 1100.0, 1100.0, 960.0, 540.0
    camera_intrinsics = torch.tensor(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )

    # Camera extrinsics are assumed to be camera-to-world in OpenGL convention; keep them on CUDA too.
    camera_extrinsics = torch.eye(4, dtype=torch.float32, device=device)
    camera_extrinsics[:3, 3] = torch.tensor([0.0, 0.0, 1.5], device=device)

    camera = Camera(
        intrinsics=camera_intrinsics,
        extrinsics=camera_extrinsics,
        convention="opengl",
        name=None,
        device=device,
    )

    # Load the point cloud onto the GPU; ensure it exposes 'rgb' and 'classification' fields.
    pc = load_point_cloud(
        filepath=str(point_cloud_path),
        device=device,
        dtype=torch.float32,
    )

    # Render RGB and segmentation; set return_mask=True when you need valid-pixel masks.
    rgb_image, rgb_valid_mask = render_rgb_from_point_cloud(
        pc=pc,
        camera=camera,
        resolution=(1080, 1920),
        return_mask=True,
        point_size=2.0,
    )

    # Segmentation rendering assumes labels live in 'classification'.
    seg_map, seg_valid_mask = render_segmentation_from_point_cloud(
        pc=pc,
        key="classification",  # Change 'key' if your labels use a different name.
        camera=camera,
        resolution=(1080, 1920),
        return_mask=True,
        point_size=2.0,
    )

    output_dir = Path("./outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_rendered_image(rgb_image, output_dir / "rgb.png")
    save_mask(rgb_valid_mask, output_dir / "rgb_valid_mask.png")
    save_mask(seg_map, output_dir / "segmentation.png")
    save_mask(seg_valid_mask, output_dir / "seg_valid_mask.png")


if __name__ == "__main__":
    demo_rendering()
