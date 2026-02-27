from typing import Optional, Tuple

import torch
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.pipelines.base_pipeline import Pipeline

from data.structures.three_d.camera.camera import Camera


@torch.no_grad()
def render_rgb_from_splatfacto(
    model: Pipeline,
    camera: Camera,
    resolution: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Render a single RGB image from a Splatfacto pipeline.

    Converts camera coordinates to OpenGL c2w convention (the pose convention expected
    by NerfStudio `Cameras`) and renders the scene from the specified camera pose.

    Args:
        model: NerfStudio pipeline containing the trained Splatfacto model.
        camera: Camera specifying intrinsics/extrinsics/convention.
        resolution: Optional (height, width) output size to render. If None, derives from intrinsics.

    Returns:
        Rendered RGB image as float32 tensor with shape (3, H, W) in range [0, 1].
    """
    assert model is not None, "NerfStudio pipeline must be provided"
    assert isinstance(camera, Camera), f"{type(camera)=}"

    device = model.model.device
    camera = camera.to(device=device, convention="opengl")

    base_cx = camera.cx
    base_cy = camera.cy
    if resolution is None:
        render_height = int(round(base_cy * 2.0))
        render_width = int(round(base_cx * 2.0))
        assert (
            render_width > 0 and render_height > 0
        ), "Unable to infer image size from intrinsics; please supply width and height explicitly"
        target_resolution = (render_height, render_width)
    else:
        target_resolution = resolution
        render_height, render_width = target_resolution

    camera_prepared = camera.scale_intrinsics(resolution=target_resolution)

    # Create camera object with intrinsic and extrinsic parameters
    ns_camera = Cameras(
        fx=camera_prepared.fx,  # Focal length in X direction
        fy=camera_prepared.fy,  # Focal length in Y direction
        cx=camera_prepared.cx,  # Principal point X
        cy=camera_prepared.cy,  # Principal point Y
        camera_to_worlds=camera_prepared.extrinsics.unsqueeze(0),
        camera_type=CameraType.PERSPECTIVE,
        width=render_width,
        height=render_height,
    )

    # Render the scene from the specified camera viewpoint
    outputs = model.model.get_outputs_for_camera(ns_camera)

    # Return RGB tensor in [3, H, W] format (transpose from [H, W, 3])
    rgb = outputs["rgb"].permute(2, 0, 1)
    assert rgb.shape == (
        3,
        render_height,
        render_width,
    ), f"Rendered output shape mismatch, expected (3, {render_height}, {render_width}), got {rgb.shape}"
    return rgb
