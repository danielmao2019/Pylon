from typing import Any, Optional, Tuple, List, Union, Dict
import copy
import torch
import torch.nn as nn
import importlib

GaussianLoDModel = importlib.import_module('utils.io.octree_gs').GaussianLoDModel
render_rgb_from_octree_gs = importlib.import_module(
    'utils.octree_gs_ops.render_rgb'
).render_rgb_from_octree_gs


def convert_model_to_density(
    model: GaussianLoDModel,
    density_color: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    uniform_scale: float = 0.02,
) -> GaussianLoDModel:
    """Convert Octree GS model to uniform density model.

    Strategy:
    - Share _anchor, _offset with original model (shallow copy)
    - Clone _scaling and keep [:, :3] unchanged (for positions), modify [:, 3:] (for visual size)
    - Monkey-patch MLP forward methods to return uniform appearance constants

    Position formula: xyz = anchor + (offset * exp(scaling[:, :3]))
    Since we keep anchor, offset, and scaling[:, :3] unchanged, positions are preserved.

    Args:
        model: Original GaussianLoDModel instance
        density_color: RGB color for density visualization (normalized 0-1)
        uniform_scale: Uniform scale value for all Gaussians

    Returns:
        Modified GaussianLoDModel with uniform appearance, preserved positions
    """
    # Create shallow copy to avoid modifying original model
    density_model = copy.copy(model)

    device = model.get_anchor.device
    num_anchors = model.get_anchor.shape[0]
    n_offsets = model.n_offsets

    # Step 1: Standardize visual scale (affects size, not position)
    # _scaling is [N, 6]: [:, :3] for position, [:, 3:] for visual size
    # We only modify [:, 3:] to standardize visual size
    modified_scaling = model._scaling.clone()
    # Set visual scale components to log(uniform_scale)
    modified_scaling[:, 3:] = torch.log(torch.tensor(uniform_scale, device=device))
    density_model._scaling = nn.Parameter(modified_scaling)

    # Step 2: Monkey-patch MLP forward methods to return uniform outputs
    # MLPs receive normal inputs but ignore them and return constants

    # 1. Opacity MLP: Always return 1.0 (full opacity)
    def constant_opacity_forward(x):
        batch_size = x.shape[0]
        return torch.ones((batch_size, n_offsets), device=x.device)
    density_model.mlp_opacity.forward = constant_opacity_forward

    # 2. Color MLP: Always return density_color
    color_tensor = torch.tensor(density_color, dtype=torch.float32)
    def constant_color_forward(x):
        batch_size = x.shape[0]
        device = x.device
        # Return [batch, 3*n_offsets]
        color_repeated = color_tensor.to(device).repeat(n_offsets)
        return color_repeated.unsqueeze(0).repeat(batch_size, 1)
    density_model.mlp_color.forward = constant_color_forward

    # 3. Cov MLP: Return uniform scale + identity rotation
    def constant_cov_forward(x):
        batch_size = x.shape[0]
        device = x.device
        # Return [batch, 7*n_offsets]: [scale_x, scale_y, scale_z, qw, qx, qy, qz]
        output = torch.zeros((batch_size, 7 * n_offsets), device=device)
        for i in range(n_offsets):
            # Scales: use large positive value to make sigmoid ≈ 1.0
            # Final visual size = exp(_scaling[:, 3:]) * sigmoid(mlp_output)
            #                   = exp(log(uniform_scale)) * sigmoid(10.0)
            #                   = uniform_scale * 0.99995
            #                   ≈ uniform_scale
            output[:, i*7:i*7+3] = 10.0  # sigmoid(10) ≈ 0.99995
            # Identity quaternion [w, x, y, z] = [1, 0, 0, 0]
            output[:, i*7+3] = 1.0  # w
            output[:, i*7+4:i*7+7] = 0.0  # x, y, z
        return output
    density_model.mlp_cov.forward = constant_cov_forward

    return density_model


@torch.no_grad()
def render_density_from_octree_gs(
    model: GaussianLoDModel,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    resolution: Optional[Tuple[int, int]] = None,
    convention: str = "opengl",
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
        model: GaussianLoDModel instance
        intrinsics: 3x3 camera intrinsics matrix
        extrinsics: 4x4 camera extrinsics matrix
        resolution: Optional (height, width) tuple for output resolution
        convention: Camera coordinate convention ('opengl', 'standard', or 'opencv')
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
    assert isinstance(model, GaussianLoDModel), f"Expected GaussianLoDModel, got {type(model)}"
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
        model=model,
        density_color=density_color,
        uniform_scale=uniform_scale
    )

    # Step 2: Use existing RGB rendering function
    result = render_rgb_from_octree_gs(
        model=density_model,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        resolution=resolution,
        convention=convention,
        background=background,
        levels=levels,
        return_info=return_info,
    )

    return result
