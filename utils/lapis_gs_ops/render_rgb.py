"""Rendering functions for LapisGS models."""

from typing import Any, Optional, Tuple, List, Union, Dict
import copy
import torch
import torch.nn as nn
import importlib

from utils.input_checks.check_camera import (
    check_camera_intrinsics,
    check_camera_extrinsics,
)

GaussianModel = importlib.import_module('utils.io.3dgs_original').GaussianModel
render_rgb_from_3dgs_original = importlib.import_module(
    'utils.3dgs_original_ops.rendering'
).render_rgb_from_3dgs_original


def filter_lapis_gs_by_layers(
    model: GaussianModel,
    layers: List[int],
    return_info: bool = False,
) -> Union[GaussianModel, Tuple[GaussianModel, Dict[str, Any]]]:
    """Filter LapisGS model to only include gaussians at specified layers.

    This function creates a filtered version of the model containing only the gaussians
    from the specified layers. Layers are defined by split_points attribute:
    - Layer 0: res8 (coarsest)
    - Layer 1: res4_only
    - Layer 2: res2_only
    - Layer 3: res1_only (finest)

    Args:
        model: GaussianModel instance with split_points and layer_names attributes
        layers: List of layers to include (each 0 to num_layers-1)
        return_info: If True, also return gaussian counts per layer

    Returns:
        If return_info is False:
            A new GaussianModel with only the gaussians at the specified layers
        If return_info is True:
            Tuple of (filtered_model, info_dict) where:
                - filtered_model: GaussianModel with only selected layers
                - info_dict: Dict containing 'gaussian_counts_per_layer' and 'total_gaussians'
    """
    # Validate model has LapisGS-specific attributes
    assert hasattr(
        model, 'split_points'
    ), "Model must have 'split_points' attribute (loaded via load_lapis_gs)"
    assert hasattr(
        model, 'layer_names'
    ), "Model must have 'layer_names' attribute (loaded via load_lapis_gs)"

    split_points = model.split_points
    layer_names = model.layer_names
    num_layers = len(split_points) - 1

    # Validate layers parameter
    assert isinstance(layers, list), f"layers must be list, got {type(layers)}"
    assert len(layers) > 0, f"layers must be non-empty, got empty list"
    for layer in layers:
        assert isinstance(layer, int), f"each layer must be int, got {type(layer)}"
        assert layer >= 0, f"each layer must be non-negative, got {layer}"
        assert (
            layer < num_layers
        ), f"layer {layer} exceeds max available layer {num_layers - 1}"

    # Count gaussians per layer if requested
    if return_info:
        gaussian_counts_per_layer = {}
        for i in range(num_layers):
            count = split_points[i + 1] - split_points[i]
            gaussian_counts_per_layer[i] = count
        total_gaussians = split_points[-1]

    # Create mask for gaussians at any of the target layers
    mask = torch.zeros(split_points[-1], dtype=torch.bool, device=model.get_xyz.device)
    for layer in layers:
        start_idx = split_points[layer]
        end_idx = split_points[layer + 1]
        mask[start_idx:end_idx] = True

    # Create a shallow copy of the model
    filtered_model = copy.copy(model)

    # Filter all the relevant attributes
    # Based on GaussianModel from utils.io.3dgs_original
    filtered_model._xyz = nn.Parameter(model._xyz[mask])
    filtered_model._features_dc = nn.Parameter(model._features_dc[mask])
    filtered_model._features_rest = nn.Parameter(model._features_rest[mask])
    filtered_model._scaling = nn.Parameter(model._scaling[mask])
    filtered_model._rotation = nn.Parameter(model._rotation[mask])
    filtered_model._opacity = nn.Parameter(model._opacity[mask])

    # Assert training-only attributes are empty, then skip filtering
    assert model.max_radii2D.numel() == 0, f"Expected max_radii2D to be empty, got shape {model.max_radii2D.shape}"
    assert model.xyz_gradient_accum.numel() == 0, f"Expected xyz_gradient_accum to be empty, got shape {model.xyz_gradient_accum.shape}"
    assert model.denom.numel() == 0, f"Expected denom to be empty, got shape {model.denom.shape}"
    # No filtering needed for empty tensors - shallow copy already handles them

    # Keep LapisGS-specific attributes (don't filter these - they describe the original structure)
    filtered_model.split_points = split_points
    filtered_model.layer_names = layer_names

    if return_info:
        info_dict = {
            'gaussian_counts_per_layer': gaussian_counts_per_layer,
            'total_gaussians': total_gaussians,
        }
        return filtered_model, info_dict
    else:
        return filtered_model


@torch.no_grad()
def render_rgb_from_lapis_gs(
    model: GaussianModel,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    resolution: Optional[Tuple[int, int]] = None,
    convention: str = "opengl",
    background: Tuple[int, int, int] = (0, 0, 0),
    layers: Optional[List[int]] = None,
    return_info: bool = False,
    device: Optional[torch.device] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    """Render RGB image from LapisGS model.

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

    Returns:
        If return_info is False:
            Rendered RGB tensor of shape (3, height, width)
        If return_info is True:
            Tuple of (RGB tensor, info_dict) where info_dict contains:
                - 'gaussian_counts_per_layer': Dict[int, int] mapping layer to gaussian count
                - 'total_gaussians': int total number of gaussians
    """
    assert isinstance(model, GaussianModel)
    check_camera_intrinsics(intrinsics)
    check_camera_extrinsics(extrinsics)

    # Validate model has LapisGS-specific attributes
    assert hasattr(
        model, 'split_points'
    ), "Model must have 'split_points' attribute (loaded via load_lapis_gs)"
    assert hasattr(
        model, 'layer_names'
    ), "Model must have 'layer_names' attribute (loaded via load_lapis_gs)"

    split_points = model.split_points
    num_layers = len(split_points) - 1

    model_device = model.get_xyz.device
    render_device = torch.device(device) if device is not None else model_device

    if model_device.type == 'cuda' and model_device.index is None:
        model_device = torch.device(f'cuda:{0}')
    if render_device.type == 'cuda' and render_device.index is None:
        render_device = torch.device(f'cuda:{model_device.index or 0}')

    if model_device != render_device:
        raise ValueError(
            "LapisGS model resides on device '{}' but rendering was requested on '{}'".format(
                model.get_xyz.device, render_device
            )
        )

    # Validate layers parameter if provided
    if layers is not None:
        assert isinstance(layers, list), f"layers must be list, got {type(layers)}"
        # If layers is empty, we'll return a background image
        for layer in layers:
            assert isinstance(layer, int), f"each layer must be int, got {type(layer)}"
            assert layer >= 0, f"each layer must be non-negative, got {layer}"
            assert (
                layer < num_layers
            ), f"layer {layer} exceeds max available layer {num_layers - 1}"

    # Handle empty layers case - return background image
    if layers is not None and len(layers) == 0:
        # Infer resolution from intrinsics if not provided
        if resolution is None:
            base_width = int(round(float(intrinsics[0, 2]) * 2.0))
            base_height = int(round(float(intrinsics[1, 2]) * 2.0))
            if base_width <= 0 or base_height <= 0:
                raise ValueError(
                    "Unable to infer image resolution from intrinsics; provide explicit resolution"
                )
            target_height = base_height
            target_width = base_width
        else:
            target_height, target_width = resolution

        background_tensor = torch.tensor(
            background, dtype=torch.float32, device=render_device
        )
        # Create a background image with the specified background color
        # Shape: (3, height, width), normalized to [0, 1] range
        background_image = (
            background_tensor.view(3, 1, 1).expand(3, target_height, target_width)
            / 255.0
        )
        return background_image

    # Handle layers filtering and info collection
    info_dict = None
    if layers is not None:
        # Filter the model to only include gaussians at the specified layers
        filter_result = filter_lapis_gs_by_layers(
            model=model,
            layers=layers,
            return_info=return_info,
        )

        if return_info:
            model_to_render, info_dict = filter_result
        else:
            model_to_render = filter_result
    else:
        # Use the original model for normal rendering
        model_to_render = model

        # If return_info=True but no filtering, still need to compute counts
        if return_info:
            gaussian_counts_per_layer = {}
            for i in range(num_layers):
                count = split_points[i + 1] - split_points[i]
                gaussian_counts_per_layer[i] = count
            total_gaussians = split_points[-1]

            info_dict = {
                'gaussian_counts_per_layer': gaussian_counts_per_layer,
                'total_gaussians': total_gaussians,
            }

    # Render using the (potentially filtered) model
    rgb = render_rgb_from_3dgs_original(
        model=model_to_render,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        resolution=resolution,
        convention=convention,
        background=background,
        device=render_device,
    )

    if return_info:
        return rgb, info_dict
    else:
        return rgb
