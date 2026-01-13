import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from data.structures.three_d.camera.camera import Camera
from models.three_d.octree_gs.loader import OctreeGS_3DGS
from models.three_d.octree_gs.model import inverse_sigmoid
from models.three_d.octree_gs.render.rgb import render_rgb_from_octree_gs


def convert_model_to_density(
    model: OctreeGS_3DGS,
    density_color: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    uniform_scale: float = 0.02,
) -> OctreeGS_3DGS:
    """Convert Octree GS model to uniform density model.

    Strategy:
    - Share _anchor, _offset with original model (shallow copy)
    - Clone _scaling and keep [:, :3] unchanged (for positions), modify [:, 3:] (for visual size)
    - Overwrite appearance tensors (_opacity, _features_dc/rest) with uniform constants

    Position formula: xyz = anchor + (offset * exp(scaling[:, :3]))
    Since we keep anchor, offset, and scaling[:, :3] unchanged, positions are preserved.

    Args:
        model: Original OctreeGS_3DGS instance
        density_color: RGB color for density visualization (normalized 0-1)
        uniform_scale: Uniform scale value for all Gaussians

    Returns:
        Modified OctreeGS_3DGS with uniform appearance, preserved positions
    """
    # Create shallow copy to avoid modifying original model
    density_model = copy.copy(model)

    device = model.get_anchor.device
    num_anchors = model.get_anchor.shape[0]

    # Step 1: Standardize visual scale (affects size, not position)
    # _scaling is [N, 6]: [:, :3] for position, [:, 3:] for visual size
    # We only modify [:, 3:] to standardize visual size
    modified_scaling = model._scaling.clone()
    # Set visual scale components to log(uniform_scale)
    scale_tensor = torch.tensor(
        uniform_scale, device=device, dtype=modified_scaling.dtype
    )
    modified_scaling[:, 3:] = torch.log(scale_tensor)
    density_model._scaling = nn.Parameter(modified_scaling)

    # Step 2: Set uniform opacity (stored in inverse-sigmoid space)
    target_opacity = torch.tensor(0.99, device=device, dtype=model._opacity.dtype)
    uniform_opacity_value = inverse_sigmoid(target_opacity)
    uniform_opacity = uniform_opacity_value.expand(num_anchors, 1).clone()
    density_model._opacity = nn.Parameter(uniform_opacity)

    # Step 3: Set uniform color via spherical harmonics DC coefficient
    # SH DC basis constant
    C0 = 0.28209479177387814
    color_tensor = torch.tensor(density_color, dtype=torch.float32, device=device)
    sh_dc_value = color_tensor / C0
    uniform_features_dc = sh_dc_value.view(1, 1, 3).expand(num_anchors, 1, 3)
    density_model._features_dc = nn.Parameter(uniform_features_dc.clone())

    # Zero higher-order SH coefficients (remove view-dependence)
    uniform_features_rest = torch.zeros(
        model._features_rest.shape,
        device=device,
        dtype=model._features_rest.dtype,
    )
    density_model._features_rest = nn.Parameter(uniform_features_rest)

    return density_model


@torch.no_grad()
def render_density_from_octree_gs(
    model: OctreeGS_3DGS,
    camera: Camera,
    resolution: Optional[Tuple[int, int]] = None,
    background: Tuple[int, int, int] = (0, 0, 0),
    levels: Optional[List[int]] = None,
    return_info: bool = False,
    density_color: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    uniform_scale: float = 0.02,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    """Render density heatmap from Octree Gaussian Splatting model.

    Visualizes spatial distribution of Gaussians with uniform appearance while
    preserving positions and level information.

    Args:
        model: OctreeGS_3DGS instance
        camera: Camera instance containing intrinsics/extrinsics/convention
        resolution: Optional (height, width) tuple for output resolution
        background: RGB background color tuple
        levels: Optional list of levels to render. If None, uses dynamic level based on camera distance.
        return_info: If True, return additional information including gaussian counts per level
        density_color: RGB color for density visualization (normalized 0-1 range)
        uniform_scale: Uniform scale value for all Gaussians

    Returns:
        If return_info is False:
            Rendered density heatmap tensor of shape (3, height, width)
        If return_info is True:
            Tuple of (density heatmap tensor, info_dict)
    """
    # Input validation
    assert isinstance(
        model, OctreeGS_3DGS
    ), f"Expected OctreeGS_3DGS, got {type(model)}"
    assert isinstance(camera, Camera), f"{type(camera)=}"
    assert (
        isinstance(density_color, tuple) and len(density_color) == 3
    ), "density_color must be an RGB tuple of 3 floats"
    assert all(
        isinstance(v, (int, float)) for v in density_color
    ), "density_color entries must be numeric"
    assert all(
        0.0 <= v <= 1.0 for v in density_color
    ), f"density_color must be in range [0, 1], got {density_color}"
    assert (
        isinstance(uniform_scale, (int, float)) and uniform_scale > 0
    ), f"uniform_scale must be positive number, got {uniform_scale}"

    # Step 1: Convert model to density model
    density_model = convert_model_to_density(
        model=model, density_color=density_color, uniform_scale=uniform_scale
    )

    # Step 2: Use existing RGB rendering function
    result = render_rgb_from_octree_gs(
        model=density_model,
        camera=camera,
        resolution=resolution,
        background=background,
        levels=levels,
        return_info=return_info,
    )

    return result
