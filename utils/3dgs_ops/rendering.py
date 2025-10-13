from typing import Optional, Tuple

import torch
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.cameras.cameras import Cameras, CameraType
from utils.three_d.camera.conventions import apply_coordinate_transform
from utils.input_checks.check_camera import (
    check_camera_intrinsics,
    check_camera_extrinsics,
)


@torch.no_grad()
def render_rgb_from_3dgs(
    model: Pipeline,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    resolution: Optional[Tuple[int, int]] = None,
    convention: str = "opengl",
) -> torch.Tensor:
    """Render a single RGB image from a Gaussian Splatting model.

    Converts camera coordinates from the specified convention to OpenGL coordinates
    and renders the scene from the specified camera pose using the provided NerfStudio pipeline.

    Args:
        model: NerfStudio pipeline containing the trained Gaussian Splatting model.
        camera_intrinsics: 3x3 camera intrinsics matrix with format [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
        camera_extrinsics: 4x4 camera-to-world transformation matrix.
        resolution: Optional (height, width) output size to render. If None, derives from intrinsics.
        convention: Camera extrinsics convention ("opengl", "standard", "opencv"). Default: "opengl".

    Returns:
        Rendered RGB image as float32 tensor with shape (3, H, W) in range [0, 1].
    """
    # Validate input parameters using utils.input_checks
    assert model is not None, "NerfStudio pipeline must be provided"
    check_camera_intrinsics(intrinsics)
    check_camera_extrinsics(extrinsics)
    assert convention in [
        "opengl",
        "standard",
        "opencv",
    ], f"convention must be 'opengl', 'standard', or 'opencv', got '{convention}'"

    device = model.model.device

    # Clone input camera_extrinsics to avoid modifying the original
    extrinsics = extrinsics.clone().to(device=device, dtype=torch.float32)

    # Extract focal lengths and principal points from intrinsics
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    # Determine target resolution and rescale intrinsics if needed
    if resolution is not None:
        render_height, render_width = resolution
    else:
        render_height = int(round(cy * 2.0))
        render_width = int(round(cx * 2.0))
        assert (
            render_width > 0 and render_height > 0
        ), "Unable to infer image size from intrinsics; please supply width and height explicitly"

    intrinsics = intrinsics.clone().to(device=device, dtype=torch.float64)
    original_width = int(intrinsics[0, 2] * 2)  # Estimate from principal point cx
    original_height = int(intrinsics[1, 2] * 2)  # Estimate from principal point cy
    assert original_width > 0, "Base width must be positive"
    assert original_height > 0, "Base height must be positive"

    scale_x = render_width / original_width
    scale_y = render_height / original_height

    intrinsics[0, 0] *= scale_x  # Scale focal length fx
    intrinsics[1, 1] *= scale_y  # Scale focal length fy
    intrinsics[0, 2] *= scale_x  # Scale principal point cx
    intrinsics[1, 2] *= scale_y  # Scale principal point cy

    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    # Convert camera extrinsics from the source convention to OpenGL convention
    # NerfStudio expects OpenGL convention for rendering
    extrinsics = apply_coordinate_transform(
        extrinsics=extrinsics, source_convention=convention, target_convention="opengl"
    )

    # Create camera object with intrinsic and extrinsic parameters
    camera = Cameras(
        fx=fx,  # Focal length in X direction
        fy=fy,  # Focal length in Y direction
        cx=cx,  # Principal point X
        cy=cy,  # Principal point Y
        camera_to_worlds=extrinsics.unsqueeze(0),
        camera_type=CameraType.PERSPECTIVE,
        width=render_width,
        height=render_height,
    )

    # Render the scene from the specified camera viewpoint
    outputs = model.model.get_outputs_for_camera(camera)

    # Return RGB tensor in [3, H, W] format (transpose from [H, W, 3])
    rgb = outputs["rgb"].permute(2, 0, 1)
    assert rgb.shape == (
        3,
        render_height,
        render_width,
    ), f"Rendered output shape mismatch, expected (3, {render_height}, {render_width}), got {rgb.shape}"
    return rgb
