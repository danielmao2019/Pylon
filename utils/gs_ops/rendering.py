import torch
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.cameras.cameras import Cameras, CameraType
from utils.three_d.camera.conventions import apply_coordinate_transform


def render_rgb_from_gs(
    pipeline: Pipeline,
    camera_intrinsics: torch.Tensor,
    camera_extrinsics: torch.Tensor,
    convention: str = "opengl",
) -> torch.Tensor:
    """Render a single RGB image from a Gaussian Splatting model.

    Converts camera coordinates from the specified convention to OpenGL coordinates
    and renders the scene from the specified camera pose using the provided pipeline.

    Args:
        pipeline: Nerfstudio pipeline containing the trained Gaussian Splatting model.
        camera_intrinsics: 3x3 camera intrinsics matrix with format [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
        camera_extrinsics: 4x4 camera-to-world transformation matrix.
        convention: Camera extrinsics convention ("opengl", "standard", "opencv"). Default: "opengl".

    Returns:
        Rendered RGB image as float32 tensor with shape (3, H, W) in range [0, 1].
    """
    # Validate input parameters
    assert pipeline is not None, "Pipeline must be provided"
    assert isinstance(camera_intrinsics, torch.Tensor), "Intrinsics must be a torch tensor"
    assert camera_intrinsics.shape == (3, 3), f"Intrinsics must be 3x3, got {camera_intrinsics.shape}"
    assert isinstance(camera_extrinsics, torch.Tensor), "Extrinsics must be a torch tensor"
    assert camera_extrinsics.shape == (4, 4), f"Extrinsics must be 4x4, got {camera_extrinsics.shape}"
    assert convention in ["opengl", "standard", "opencv"], f"convention must be 'opengl', 'standard', or 'opencv', got '{convention}'"
    
    # Clone input camera_extrinsics to avoid modifying the original
    camera_extrinsics = camera_extrinsics.clone()
    
    # Extract focal lengths and principal points from intrinsics
    fx = float(camera_intrinsics[0, 0])
    fy = float(camera_intrinsics[1, 1])
    cx = float(camera_intrinsics[0, 2])
    cy = float(camera_intrinsics[1, 2])
    
    # Calculate image dimensions from principal points (assuming principal point is at image center)
    image_width = int(cx * 2)
    image_height = int(cy * 2)
    
    # Convert camera extrinsics from the source convention to OpenGL convention
    # Nerfstudio expects OpenGL convention for rendering
    camera_extrinsics = apply_coordinate_transform(
        camera_extrinsics=camera_extrinsics,
        source_convention=convention,
        target_convention="opengl"
    )

    # Create camera object with intrinsic and extrinsic parameters
    camera = Cameras(
        fx=fx,  # Focal length in X direction
        fy=fy,  # Focal length in Y direction
        cx=cx,  # Principal point X
        cy=cy,  # Principal point Y
        camera_to_worlds=camera_extrinsics.unsqueeze(0),
        camera_type=CameraType.PERSPECTIVE,
        width=image_width,
        height=image_height,
    )
    
    # Render the scene from the specified camera viewpoint
    outputs = pipeline.model.get_outputs_for_camera(camera)
    
    # Return RGB tensor in [3, H, W] format (transpose from [H, W, 3])
    outputs = outputs["rgb"].permute(2, 0, 1)
    assert outputs.shape == (3, image_height, image_width), f"Rendered output shape mismatch, expected (3, {image_height}, {image_width}), got {outputs.shape}"
    return outputs
