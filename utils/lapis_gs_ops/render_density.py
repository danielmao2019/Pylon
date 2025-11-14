"""Density rendering functions for LapisGS models."""

from typing import Any, Optional, Tuple, List, Union, Dict
import copy
import math
import torch
import torch.nn as nn
import importlib

from utils.input_checks.check_camera import (
    check_camera_intrinsics,
    check_camera_extrinsics,
)

GaussianModel = importlib.import_module('utils.io.3dgs_original').GaussianModel
inverse_sigmoid = importlib.import_module('utils.io.3dgs_original').inverse_sigmoid
render_rgb_from_lapis_gs = importlib.import_module(
    'utils.lapis_gs_ops.render_rgb'
).render_rgb_from_lapis_gs


def convert_model_to_density(
    model: GaussianModel,
    density_color: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    uniform_scale: float = 0.02,
) -> GaussianModel:
    """Convert LapisGS model to uniform density model.

    Strategy:
    - Share _xyz and _rotation with original model (shallow copy)
    - Modify _opacity to uniform high opacity
    - Modify _features_dc to uniform density_color
    - Set _features_rest to zeros (no view-dependent color)
    - Modify _scaling to uniform scale

    Since we keep _xyz and _rotation unchanged, positions and orientations are preserved.

    Args:
        model: GaussianModel instance with split_points and layer_names attributes
        density_color: RGB color for density visualization (normalized 0-1 range)
        uniform_scale: Uniform scale value for all Gaussians

    Returns:
        Modified GaussianModel with uniform appearance, preserved positions
    """
    # Create shallow copy to avoid modifying original model
    density_model = copy.copy(model)

    device = model.get_xyz.device
    num_gaussians = model.get_xyz.shape[0]

    # Step 1: Set uniform high opacity
    # _opacity stores inverse_sigmoid values, so we set to inverse_sigmoid(0.99)
    target_opacity = 0.99
    uniform_opacity = torch.full(
        (num_gaussians, 1),
        inverse_sigmoid(torch.tensor(target_opacity, device=device)),
        device=device,
    )
    density_model._opacity = nn.Parameter(uniform_opacity)

    # Step 2: Set uniform color via spherical harmonics DC coefficient
    # The DC coefficient C0 = 1/(2*sqrt(pi)) ≈ 0.28209
    # To get RGB color: rgb = SH_DC * C0
    # So: SH_DC = rgb / C0
    C0 = 0.28209479177387814  # SH basis constant for degree 0
    color_tensor = torch.tensor(density_color, dtype=torch.float32, device=device)
    sh_dc_value = color_tensor / C0
    # _features_dc shape: (N, 1, 3) where second dim is SH degree 0
    uniform_features_dc = sh_dc_value.view(1, 1, 3).expand(num_gaussians, 1, 3)
    density_model._features_dc = nn.Parameter(uniform_features_dc.clone())

    # Step 3: Zero out higher-order SH coefficients (no view-dependent color)
    sh_rest_shape = model._features_rest.shape  # (N, 15, 3) for max_sh_degree=3
    uniform_features_rest = torch.zeros(sh_rest_shape, device=device)
    density_model._features_rest = nn.Parameter(uniform_features_rest)

    # Step 4: Set uniform scale
    # _scaling stores log(scale), so we set to log(uniform_scale)
    log_scale = math.log(uniform_scale)
    uniform_scaling = torch.full(
        (num_gaussians, 3), log_scale, device=device, dtype=torch.float32
    )
    density_model._scaling = nn.Parameter(uniform_scaling)

    # Keep LapisGS-specific attributes (don't filter these - they describe the original structure)
    density_model.split_points = model.split_points
    density_model.layer_names = model.layer_names

    return density_model


@torch.no_grad()
def render_density_from_lapis_gs(
    model: GaussianModel,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    resolution: Optional[Tuple[int, int]] = None,
    convention: str = "opengl",
    background: Tuple[int, int, int] = (0, 0, 0),
    layers: Optional[List[int]] = None,
    return_info: bool = False,
    density_color: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    uniform_scale: float = 0.02,
    device: Optional[torch.device] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    """Render density heatmap from LapisGS model.

    Visualizes spatial distribution of Gaussians with uniform appearance while
    preserving positions and layer information.

    Args:
        model: GaussianModel instance with split_points and layer_names attributes
        intrinsics: 3x3 camera intrinsics matrix
        extrinsics: 4x4 camera extrinsics matrix
        resolution: Optional (height, width) tuple for output resolution
        convention: Camera coordinate convention ('opengl', 'standard', or 'opencv')
        background: RGB background color tuple
        layers: Optional list of layers to render. If None, uses all layers (default).
                If provided, renders only Gaussians at the specified layers (each 0 to num_layers-1).
                Can be empty list to render only background.
        return_info: If True, return additional information including gaussian counts per layer
        density_color: RGB color for density visualization (normalized 0-1 range)
        uniform_scale: Uniform scale value for all Gaussians

    Returns:
        If return_info is False:
            Rendered density heatmap tensor of shape (3, height, width)
        If return_info is True:
            Tuple of (density heatmap tensor, info_dict) where info_dict contains:
                - 'gaussian_counts_per_layer': Dict[int, int] mapping layer to gaussian count
                - 'total_gaussians': int total number of gaussians
    """
    # Input validation
    assert isinstance(model, GaussianModel), f"Expected GaussianModel, got {type(model)}"
    check_camera_intrinsics(intrinsics)
    check_camera_extrinsics(extrinsics)

    # Validate model has LapisGS-specific attributes
    assert hasattr(
        model, 'split_points'
    ), "Model must have 'split_points' attribute (loaded via load_lapis_gs)"
    assert hasattr(
        model, 'layer_names'
    ), "Model must have 'layer_names' attribute (loaded via load_lapis_gs)"

    assert isinstance(density_color, tuple) and len(density_color) == 3, \
        "density_color must be an RGB tuple of 3 floats"
    assert all(isinstance(v, (int, float)) for v in density_color), \
        "density_color entries must be numeric"
    assert all(0.0 <= v <= 1.0 for v in density_color), \
        f"density_color must be in range [0, 1], got {density_color}"
    assert isinstance(uniform_scale, (int, float)) and uniform_scale > 0, \
        f"uniform_scale must be positive number, got {uniform_scale}"

    # Step 1: Convert model to density model
    density_model = convert_model_to_density(
        model=model, density_color=density_color, uniform_scale=uniform_scale
    )

    # Step 2: Use existing RGB rendering function
    result = render_rgb_from_lapis_gs(
        model=density_model,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        resolution=resolution,
        convention=convention,
        background=background,
        layers=layers,
        return_info=return_info,
        device=device,
    )

    return result
