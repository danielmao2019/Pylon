"""Dash pixel display APIs."""

from typing import Dict, Tuple

import torch
from dash import dcc

from data.viewer.utils.displays.pixels.dash.core_pixels_display import (
    create_dash_pixels_display,
    create_image_display,
    create_segmentation_display,
    get_image_display_stats,
    get_segmentation_display_stats,
)
from data.viewer.utils.displays.pixels.dash.depth_image_display import (
    create_depth_display,
    get_depth_display_stats,
)
from data.viewer.utils.displays.pixels.dash.edge_image_display import (
    create_edge_display,
    get_edge_display_stats,
)
from data.viewer.utils.displays.pixels.dash.instance_surrogate_image_display import (
    create_instance_surrogate_display,
    get_instance_surrogate_display_stats,
)
from data.viewer.utils.displays.pixels.dash.normal_image_display import (
    create_normal_display,
    get_normal_display_stats,
)
from data.viewer.utils.displays.utils.class_colors import map_class_ids_to_rgb
from data.viewer.utils.displays.utils.heatmap_colors import map_scalars_to_rgb
from utils.io.image import load_image

__all__ = [
    "create_depth_display",
    "create_edge_display",
    "create_image_display",
    "create_instance_surrogate_display",
    "create_normal_display",
    "create_segmentation_display",
    "get_depth_display_stats",
    "get_edge_display_stats",
    "get_image_display_stats",
    "get_instance_surrogate_display_stats",
    "get_normal_display_stats",
    "get_segmentation_display_stats",
]

DEFAULT_COLOR_IMAGE_INTERPOLATION = "linear"
DEFAULT_DEPTH_IMAGE_INTERPOLATION = "nearest"
DEFAULT_EDGE_IMAGE_INTERPOLATION = "nearest"
DEFAULT_NORMAL_IMAGE_INTERPOLATION = "nearest"
DEFAULT_SEGMENTATION_IMAGE_INTERPOLATION = "nearest"
DEFAULT_INSTANCE_SURROGATE_IMAGE_INTERPOLATION = "nearest"


def create_color_image_display(
    color_image_path: str,
    image_interpolation: str = DEFAULT_COLOR_IMAGE_INTERPOLATION,
) -> dcc.Graph:
    """Render a color image display from an image artifact path.

    Args:
        color_image_path: Filesystem path to the color image artifact.
        image_interpolation: Interpolation choice forwarded to the Dash figure.

    Returns:
        Dash ``dcc.Graph`` rendering the color image.
    """
    assert isinstance(
        color_image_path, str
    ), "Color image path must be a string. color_image_path=%r" % (color_image_path,)
    assert isinstance(
        image_interpolation, str
    ), "Image interpolation must be a string. image_interpolation=%r" % (
        image_interpolation,
    )

    color_image = load_image(filepath=color_image_path, normalization=None)
    rgb_image = color_image.permute(1, 2, 0).to(torch.uint8)
    return create_dash_pixels_display(
        image=rgb_image,
        image_interpolation=image_interpolation,
    )


def create_depth_image_display(
    depth_image_path: str,
    image_interpolation: str = DEFAULT_DEPTH_IMAGE_INTERPOLATION,
) -> dcc.Graph:
    """Render a depth image display from an image artifact path.

    Args:
        depth_image_path: Filesystem path to the depth image artifact.
        image_interpolation: Interpolation choice forwarded to the Dash figure.

    Returns:
        Dash ``dcc.Graph`` rendering the colorized depth image.
    """
    assert isinstance(
        depth_image_path, str
    ), "Depth image path must be a string. depth_image_path=%r" % (depth_image_path,)
    assert isinstance(
        image_interpolation, str
    ), "Image interpolation must be a string. image_interpolation=%r" % (
        image_interpolation,
    )

    rgb_image = _map_depth_image_to_rgb(depth_image_path=depth_image_path)
    return create_dash_pixels_display(
        image=rgb_image,
        image_interpolation=image_interpolation,
    )


def create_edge_image_display(
    edge_image_path: str,
    image_interpolation: str = DEFAULT_EDGE_IMAGE_INTERPOLATION,
) -> dcc.Graph:
    """Render an edge image display from an image artifact path.

    Args:
        edge_image_path: Filesystem path to the edge image artifact.
        image_interpolation: Interpolation choice forwarded to the Dash figure.

    Returns:
        Dash ``dcc.Graph`` rendering the grayscale edge image.
    """
    assert isinstance(
        edge_image_path, str
    ), "Edge image path must be a string. edge_image_path=%r" % (edge_image_path,)
    assert isinstance(
        image_interpolation, str
    ), "Image interpolation must be a string. image_interpolation=%r" % (
        image_interpolation,
    )

    rgb_image = _map_edge_image_to_rgb(edge_image_path=edge_image_path)
    return create_dash_pixels_display(
        image=rgb_image,
        image_interpolation=image_interpolation,
    )


def create_normal_image_display(
    normal_image_path: str,
    image_interpolation: str = DEFAULT_NORMAL_IMAGE_INTERPOLATION,
) -> dcc.Graph:
    """Render a surface-normal image display from an image artifact path.

    Args:
        normal_image_path: Filesystem path to the normal image artifact.
        image_interpolation: Interpolation choice forwarded to the Dash figure.

    Returns:
        Dash ``dcc.Graph`` rendering the colorized normal image.
    """
    assert isinstance(
        normal_image_path, str
    ), "Normal image path must be a string. normal_image_path=%r" % (normal_image_path,)
    assert isinstance(
        image_interpolation, str
    ), "Image interpolation must be a string. image_interpolation=%r" % (
        image_interpolation,
    )

    rgb_image = _map_normal_image_to_rgb(normal_image_path=normal_image_path)
    return create_dash_pixels_display(
        image=rgb_image,
        image_interpolation=image_interpolation,
    )


def create_segmentation_image_display(
    segmentation_image_path: str,
    image_interpolation: str = DEFAULT_SEGMENTATION_IMAGE_INTERPOLATION,
) -> dcc.Graph:
    """Render the backend-colorized segmentation image display.

    Args:
        segmentation_image_path: Filesystem path to the segmentation image artifact.
        image_interpolation: Interpolation choice forwarded to the Dash figure.

    Returns:
        Dash ``dcc.Graph`` rendering the class-colorized segmentation image.
    """
    assert isinstance(
        segmentation_image_path, str
    ), "Segmentation image path must be a string. segmentation_image_path=%r" % (
        segmentation_image_path,
    )
    assert isinstance(
        image_interpolation, str
    ), "Image interpolation must be a string. image_interpolation=%r" % (
        image_interpolation,
    )

    segmentation_image = load_image(
        filepath=segmentation_image_path,
        normalization=None,
    ).to(torch.int64)
    class_id_to_rgb = map_class_ids_to_rgb(class_ids=torch.unique(segmentation_image))
    rgb_image = _map_segmentation_image_to_rgb(
        segmentation_image_path=segmentation_image_path,
        class_id_to_rgb=class_id_to_rgb,
    )
    return create_dash_pixels_display(
        image=rgb_image,
        image_interpolation=image_interpolation,
    )


def create_instance_surrogate_image_display(
    image_path: str,
    image_interpolation: str = DEFAULT_INSTANCE_SURROGATE_IMAGE_INTERPOLATION,
) -> dcc.Graph:
    """Render the backend-colorized instance-surrogate image display.

    Args:
        image_path: Filesystem path to the instance-surrogate image artifact.
        image_interpolation: Interpolation choice forwarded to the Dash figure.

    Returns:
        Dash ``dcc.Graph`` rendering the class-colorized instance-surrogate image.
    """
    assert isinstance(
        image_path, str
    ), "Instance-surrogate image path must be a string. image_path=%r" % (image_path,)
    assert isinstance(
        image_interpolation, str
    ), "Image interpolation must be a string. image_interpolation=%r" % (
        image_interpolation,
    )

    instance_surrogate = load_image(filepath=image_path, normalization=None)
    assert instance_surrogate.ndim == 3 and instance_surrogate.shape[0] >= 2, (
        "Instance-surrogate image must have at least 2 channels. "
        "instance_surrogate.shape=%r" % (instance_surrogate.shape,)
    )
    y_offset = instance_surrogate[0].to(torch.float64)
    x_offset = instance_surrogate[1].to(torch.float64)
    magnitude = torch.sqrt(y_offset**2 + x_offset**2)
    instance_surrogate_class_id_image = torch.zeros_like(magnitude, dtype=torch.int64)
    percentiles = torch.quantile(
        magnitude.reshape(-1),
        torch.linspace(0, 1, 20, dtype=torch.float64),
    )
    for bin_index in range(len(percentiles) - 1):
        if bin_index == len(percentiles) - 2:
            mask = magnitude >= percentiles[bin_index]
        else:
            mask = (magnitude >= percentiles[bin_index]) & (
                magnitude < percentiles[bin_index + 1]
            )
        instance_surrogate_class_id_image[mask] = bin_index + 1
    class_id_to_rgb = map_class_ids_to_rgb(
        class_ids=torch.unique(instance_surrogate_class_id_image),
    )
    rgb_image = _map_instance_surrogate_image_to_rgb(
        image_path=image_path,
        class_id_to_rgb=class_id_to_rgb,
    )
    return create_dash_pixels_display(
        image=rgb_image,
        image_interpolation=image_interpolation,
    )


def _map_depth_image_to_rgb(depth_image_path: str) -> torch.Tensor:
    """Decode a depth image into a colorized RGB tensor via a heatmap palette.

    Args:
        depth_image_path: Filesystem path to the depth image artifact.

    Returns:
        RGB tensor of shape ``[H, W, 3]``, uint8 dtype, channel values in ``[0, 255]``.
    """
    assert isinstance(
        depth_image_path, str
    ), "Depth image path must be a string. depth_image_path=%r" % (depth_image_path,)

    depth_image = load_image(filepath=depth_image_path, normalization=None)
    if depth_image.ndim == 3:
        depth_image = depth_image[0]
    depth_scalars = depth_image.to(torch.float64)
    depth_scalars = depth_scalars - float(depth_scalars.min().item())
    return map_scalars_to_rgb(scalars=depth_scalars)


def _map_edge_image_to_rgb(edge_image_path: str) -> torch.Tensor:
    """Decode an edge image into a grayscale RGB tensor.

    Args:
        edge_image_path: Filesystem path to the edge image artifact.

    Returns:
        RGB tensor of shape ``[H, W, 3]``, uint8 dtype, channel values in ``[0, 255]``.
    """
    assert isinstance(
        edge_image_path, str
    ), "Edge image path must be a string. edge_image_path=%r" % (edge_image_path,)

    edge_image = load_image(filepath=edge_image_path, normalization=None)
    if edge_image.ndim == 3:
        edge_image = edge_image[0]
    edge_float = edge_image.to(torch.float64)
    edge_min = float(edge_float.min().item())
    edge_max = float(edge_float.max().item())
    normalized = (edge_float - edge_min) / max(edge_max - edge_min, 1e-12)
    gray = (normalized * 255.0).round().clamp(min=0.0, max=255.0).to(torch.uint8)
    return gray.unsqueeze(-1).repeat(1, 1, 3)


def _map_normal_image_to_rgb(normal_image_path: str) -> torch.Tensor:
    """Decode a surface-normal image into an RGB tensor.

    Maps normal vector components from ``[-1, 1]`` to ``[0, 255]`` per channel,
    matching the live tensor-based normal colorization convention.

    Args:
        normal_image_path: Filesystem path to the normal image artifact.

    Returns:
        RGB tensor of shape ``[H, W, 3]``, uint8 dtype, channel values in ``[0, 255]``.
    """
    assert isinstance(
        normal_image_path, str
    ), "Normal image path must be a string. normal_image_path=%r" % (normal_image_path,)

    normal_image = load_image(filepath=normal_image_path, normalization=None)
    normal_float = normal_image.to(torch.float64)
    normal_float = normal_float / 127.5 - 1.0
    normals_normalized = (normal_float + 1.0) / 2.0
    normals_normalized = normals_normalized.clamp(min=0.0, max=1.0)
    rgb = (normals_normalized * 255.0).round().clamp(min=0.0, max=255.0).to(torch.uint8)
    return rgb.permute(1, 2, 0)


def _map_segmentation_image_to_rgb(
    segmentation_image_path: str,
    class_id_to_rgb: Dict[int, Tuple[int, int, int]],
) -> torch.Tensor:
    """Decode a segmentation image into a class-colorized RGB tensor.

    Args:
        segmentation_image_path: Filesystem path to the segmentation image artifact.
        class_id_to_rgb: Mapping from class identifier to RGB color tuple with
            channel values in ``[0, 255]``.

    Returns:
        RGB tensor of shape ``[H, W, 3]``, uint8 dtype, channel values in ``[0, 255]``.
    """
    assert isinstance(
        segmentation_image_path, str
    ), "Segmentation image path must be a string. segmentation_image_path=%r" % (
        segmentation_image_path,
    )
    assert isinstance(
        class_id_to_rgb, dict
    ), "Class-id to RGB mapping must be a dict. class_id_to_rgb=%r" % (class_id_to_rgb,)

    segmentation_image = load_image(
        filepath=segmentation_image_path,
        normalization=None,
    ).to(torch.int64)
    assert (
        segmentation_image.ndim == 2
    ), "Segmentation image must have shape [H, W]. segmentation_image.shape=%r" % (
        segmentation_image.shape,
    )
    height, width = segmentation_image.shape
    rgb_image = torch.zeros((height, width, 3), dtype=torch.uint8)
    for class_id, color in class_id_to_rgb.items():
        rgb_image[segmentation_image == class_id] = torch.tensor(
            color,
            dtype=torch.uint8,
        )
    return rgb_image


def _map_instance_surrogate_image_to_rgb(
    image_path: str,
    class_id_to_rgb: Dict[int, Tuple[int, int, int]],
) -> torch.Tensor:
    """Decode an instance-surrogate image into a class-colorized RGB tensor.

    Args:
        image_path: Filesystem path to the instance-surrogate image artifact.
        class_id_to_rgb: Mapping from class identifier to RGB color tuple with
            channel values in ``[0, 255]``.

    Returns:
        RGB tensor of shape ``[H, W, 3]``, uint8 dtype, channel values in ``[0, 255]``.
    """
    assert isinstance(
        image_path, str
    ), "Instance-surrogate image path must be a string. image_path=%r" % (image_path,)
    assert isinstance(
        class_id_to_rgb, dict
    ), "Class-id to RGB mapping must be a dict. class_id_to_rgb=%r" % (class_id_to_rgb,)

    instance_surrogate = load_image(filepath=image_path, normalization=None)
    assert instance_surrogate.ndim == 3 and instance_surrogate.shape[0] >= 2, (
        "Instance-surrogate image must have at least 2 channels. "
        "instance_surrogate.shape=%r" % (instance_surrogate.shape,)
    )
    y_offset = instance_surrogate[0].to(torch.float64)
    x_offset = instance_surrogate[1].to(torch.float64)
    magnitude = torch.sqrt(y_offset**2 + x_offset**2)
    class_id_image = torch.zeros_like(magnitude, dtype=torch.int64)
    percentiles = torch.quantile(
        magnitude.reshape(-1),
        torch.linspace(0, 1, 20, dtype=torch.float64),
    )
    for bin_index in range(len(percentiles) - 1):
        if bin_index == len(percentiles) - 2:
            mask = magnitude >= percentiles[bin_index]
        else:
            mask = (magnitude >= percentiles[bin_index]) & (
                magnitude < percentiles[bin_index + 1]
            )
        class_id_image[mask] = bin_index + 1

    height, width = class_id_image.shape
    rgb_image = torch.zeros((height, width, 3), dtype=torch.uint8)
    for class_id, color in class_id_to_rgb.items():
        rgb_image[class_id_image == class_id] = torch.tensor(
            color,
            dtype=torch.uint8,
        )
    return rgb_image
