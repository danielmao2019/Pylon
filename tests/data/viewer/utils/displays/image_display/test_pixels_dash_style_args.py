"""Tests for the branch's new pixels Dash style-args functions - Valid Cases.

Covers ``create_dash_pixels_display`` (core), the six ``create_*_image_display``
path-based APIs and their ``DEFAULT_*_INTERPOLATION`` defaults, plus the
``_map_*_image_to_rgb`` helpers.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""

import numpy as np
import torch
from dash import dcc
from PIL import Image

from data.viewer.utils.displays.pixels.dash.apis import (
    DEFAULT_COLOR_IMAGE_INTERPOLATION,
    DEFAULT_DEPTH_IMAGE_INTERPOLATION,
    DEFAULT_EDGE_IMAGE_INTERPOLATION,
    DEFAULT_INSTANCE_SURROGATE_IMAGE_INTERPOLATION,
    DEFAULT_NORMAL_IMAGE_INTERPOLATION,
    DEFAULT_SEGMENTATION_IMAGE_INTERPOLATION,
    _map_depth_image_to_rgb,
    _map_segmentation_image_to_rgb,
    create_color_image_display,
    create_depth_image_display,
    create_edge_image_display,
    create_instance_surrogate_image_display,
    create_normal_image_display,
    create_segmentation_image_display,
)
from data.viewer.utils.displays.pixels.dash.core_pixels_display import (
    create_dash_pixels_display,
)
from data.viewer.utils.displays.utils.class_colors import get_class_color

# ================================================================================
# Helpers: synthesize on-disk image fixtures for ``utils.io.image.load_image``.
#
# ``load_image`` decodes via PIL:
#   - mode 'L'/'P' PNG   -> [H, W]    uint8
#   - mode 'RGB' PNG     -> [3, H, W] uint8
# ================================================================================


def _save_gray_png(tmp_path, name, array_hw_uint8):
    """Save an [H, W] uint8 numpy array as a grayscale PNG and return its path."""
    path = tmp_path / name
    Image.fromarray(array_hw_uint8, mode="L").save(str(path))
    return str(path)


def _save_rgb_png(tmp_path, name, array_hwc_uint8):
    """Save an [H, W, 3] uint8 numpy array as an RGB PNG and return its path."""
    path = tmp_path / name
    Image.fromarray(array_hwc_uint8, mode="RGB").save(str(path))
    return str(path)


def _trace_zsmooth(graph):
    """Extract the single Image trace's ``zsmooth`` value from a dcc.Graph."""
    assert isinstance(graph, dcc.Graph)
    figure = graph.figure
    assert len(figure.data) == 1
    return figure.data[0].zsmooth


# ================================================================================
# create_dash_pixels_display Tests
# ================================================================================


def test_create_dash_pixels_display_returns_graph():
    """Resolved RGB array produces a dcc.Graph carrying an Image trace."""
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    graph = create_dash_pixels_display(image=image, image_interpolation="nearest")
    assert isinstance(graph, dcc.Graph)
    assert len(graph.figure.data) == 1
    assert graph.figure.data[0].type == "image"


def test_create_dash_pixels_display_nearest_disables_smoothing():
    """``nearest`` maps to zsmooth=False (no smoothing)."""
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    graph = create_dash_pixels_display(image=image, image_interpolation="nearest")
    assert _trace_zsmooth(graph) is False


def test_create_dash_pixels_display_non_nearest_enables_smoothing():
    """A non-``nearest`` value enables smoothing (zsmooth='fast')."""
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    graph = create_dash_pixels_display(image=image, image_interpolation="linear")
    assert _trace_zsmooth(graph) == "fast"


def test_create_dash_pixels_display_accepts_tensor():
    """A torch.Tensor RGB image is accepted and forwarded to the figure."""
    image = torch.zeros((4, 5, 3), dtype=torch.uint8)
    graph = create_dash_pixels_display(image=image, image_interpolation="bilinear")
    assert isinstance(graph, dcc.Graph)
    assert _trace_zsmooth(graph) == "fast"


# ================================================================================
# create_color_image_display Tests
# ================================================================================


def test_create_color_image_display_default_interpolation(tmp_path):
    """Color display applies DEFAULT_COLOR_IMAGE_INTERPOLATION ('linear' -> 'fast')."""
    assert DEFAULT_COLOR_IMAGE_INTERPOLATION == "linear"
    array = np.full((6, 7, 3), 128, dtype=np.uint8)
    path = _save_rgb_png(tmp_path, "color.png", array)
    graph = create_color_image_display(color_image_path=path)
    assert isinstance(graph, dcc.Graph)
    # Default 'linear' is not 'nearest' -> smoothing enabled.
    assert _trace_zsmooth(graph) == "fast"


def test_create_color_image_display_explicit_override(tmp_path):
    """Explicit 'nearest' overrides the default and reaches the figure."""
    array = np.full((6, 7, 3), 128, dtype=np.uint8)
    path = _save_rgb_png(tmp_path, "color.png", array)
    graph = create_color_image_display(
        color_image_path=path, image_interpolation="nearest"
    )
    assert _trace_zsmooth(graph) is False


def test_create_color_image_display_pixels_roundtrip(tmp_path):
    """The figure's z buffer matches the saved RGB pixels."""
    array = np.zeros((4, 4, 3), dtype=np.uint8)
    array[0, 0] = (10, 20, 30)
    array[1, 2] = (200, 100, 50)
    path = _save_rgb_png(tmp_path, "color.png", array)
    graph = create_color_image_display(color_image_path=path)
    z = np.asarray(graph.figure.data[0].z)
    assert z.shape == (4, 4, 3)
    assert tuple(z[0, 0]) == (10, 20, 30)
    assert tuple(z[1, 2]) == (200, 100, 50)


# ================================================================================
# create_depth_image_display Tests
# ================================================================================


def test_create_depth_image_display_default_interpolation(tmp_path):
    """Depth display applies DEFAULT_DEPTH_IMAGE_INTERPOLATION ('nearest' -> False)."""
    assert DEFAULT_DEPTH_IMAGE_INTERPOLATION == "nearest"
    array = np.array([[0, 255], [64, 128]], dtype=np.uint8)
    path = _save_gray_png(tmp_path, "depth.png", array)
    graph = create_depth_image_display(depth_image_path=path)
    assert isinstance(graph, dcc.Graph)
    assert _trace_zsmooth(graph) is False


def test_create_depth_image_display_explicit_override(tmp_path):
    """Explicit 'linear' overrides the 'nearest' default and reaches the figure."""
    array = np.array([[0, 255], [64, 128]], dtype=np.uint8)
    path = _save_gray_png(tmp_path, "depth.png", array)
    graph = create_depth_image_display(
        depth_image_path=path, image_interpolation="linear"
    )
    assert _trace_zsmooth(graph) == "fast"


# ================================================================================
# create_edge_image_display Tests
# ================================================================================


def test_create_edge_image_display_default_interpolation(tmp_path):
    """Edge display applies DEFAULT_EDGE_IMAGE_INTERPOLATION ('nearest' -> False)."""
    assert DEFAULT_EDGE_IMAGE_INTERPOLATION == "nearest"
    array = np.array([[0, 255], [10, 200]], dtype=np.uint8)
    path = _save_gray_png(tmp_path, "edge.png", array)
    graph = create_edge_image_display(edge_image_path=path)
    assert isinstance(graph, dcc.Graph)
    assert _trace_zsmooth(graph) is False


def test_create_edge_image_display_explicit_override(tmp_path):
    """Explicit 'linear' overrides the default and reaches the figure."""
    array = np.array([[0, 255], [10, 200]], dtype=np.uint8)
    path = _save_gray_png(tmp_path, "edge.png", array)
    graph = create_edge_image_display(
        edge_image_path=path, image_interpolation="linear"
    )
    assert _trace_zsmooth(graph) == "fast"


def test_create_edge_image_display_grayscale_extremes(tmp_path):
    """Edge min pixel maps to black (0,0,0) and max pixel to white (255,255,255)."""
    array = np.array([[0, 255], [10, 200]], dtype=np.uint8)
    path = _save_gray_png(tmp_path, "edge.png", array)
    graph = create_edge_image_display(edge_image_path=path)
    z = np.asarray(graph.figure.data[0].z)
    assert tuple(z[0, 0]) == (0, 0, 0)
    assert tuple(z[0, 1]) == (255, 255, 255)


# ================================================================================
# create_normal_image_display Tests
# ================================================================================


def test_create_normal_image_display_default_interpolation(tmp_path):
    """Normal display applies DEFAULT_NORMAL_IMAGE_INTERPOLATION ('nearest' -> False)."""
    assert DEFAULT_NORMAL_IMAGE_INTERPOLATION == "nearest"
    array = np.full((4, 5, 3), 200, dtype=np.uint8)
    path = _save_rgb_png(tmp_path, "normal.png", array)
    graph = create_normal_image_display(normal_image_path=path)
    assert isinstance(graph, dcc.Graph)
    assert _trace_zsmooth(graph) is False


def test_create_normal_image_display_explicit_override(tmp_path):
    """Explicit 'linear' overrides the default and reaches the figure."""
    array = np.full((4, 5, 3), 200, dtype=np.uint8)
    path = _save_rgb_png(tmp_path, "normal.png", array)
    graph = create_normal_image_display(
        normal_image_path=path, image_interpolation="linear"
    )
    assert _trace_zsmooth(graph) == "fast"


# ================================================================================
# create_segmentation_image_display Tests
# ================================================================================


def test_create_segmentation_image_display_default_interpolation(tmp_path):
    """Segmentation applies DEFAULT_SEGMENTATION_IMAGE_INTERPOLATION ('nearest')."""
    assert DEFAULT_SEGMENTATION_IMAGE_INTERPOLATION == "nearest"
    array = np.array([[0, 1], [2, 0]], dtype=np.uint8)
    path = _save_gray_png(tmp_path, "seg.png", array)
    graph = create_segmentation_image_display(segmentation_image_path=path)
    assert isinstance(graph, dcc.Graph)
    assert _trace_zsmooth(graph) is False


def test_create_segmentation_image_display_explicit_override(tmp_path):
    """Explicit 'linear' overrides the default and reaches the figure."""
    array = np.array([[0, 1], [2, 0]], dtype=np.uint8)
    path = _save_gray_png(tmp_path, "seg.png", array)
    graph = create_segmentation_image_display(
        segmentation_image_path=path, image_interpolation="linear"
    )
    assert _trace_zsmooth(graph) == "fast"


def test_create_segmentation_image_display_class_colors(tmp_path):
    """Each pixel's color matches the class-id palette color for its class."""
    array = np.array([[0, 1], [2, 0]], dtype=np.uint8)
    path = _save_gray_png(tmp_path, "seg.png", array)
    graph = create_segmentation_image_display(segmentation_image_path=path)
    z = np.asarray(graph.figure.data[0].z)
    assert tuple(z[0, 0]) == get_class_color(class_id=0)
    assert tuple(z[0, 1]) == get_class_color(class_id=1)
    assert tuple(z[1, 0]) == get_class_color(class_id=2)
    assert tuple(z[1, 1]) == get_class_color(class_id=0)


# ================================================================================
# create_instance_surrogate_image_display Tests
# ================================================================================


def test_create_instance_surrogate_image_display_default_interpolation(tmp_path):
    """Instance-surrogate applies its DEFAULT ('nearest' -> False)."""
    assert DEFAULT_INSTANCE_SURROGATE_IMAGE_INTERPOLATION == "nearest"
    # 3-channel RGB PNG -> load_image gives [3, H, W]; >= 2 channels required.
    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    path = _save_rgb_png(tmp_path, "inst.png", array)
    graph = create_instance_surrogate_image_display(image_path=path)
    assert isinstance(graph, dcc.Graph)
    assert _trace_zsmooth(graph) is False


def test_create_instance_surrogate_image_display_explicit_override(tmp_path):
    """Explicit 'linear' overrides the default and reaches the figure."""
    rng = np.random.default_rng(1)
    array = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    path = _save_rgb_png(tmp_path, "inst.png", array)
    graph = create_instance_surrogate_image_display(
        image_path=path, image_interpolation="linear"
    )
    assert _trace_zsmooth(graph) == "fast"


# ================================================================================
# _map_*_image_to_rgb Helper Tests
# ================================================================================


def test_map_depth_image_to_rgb_shape_dtype_and_endpoints(tmp_path):
    """Depth helper yields [H, W, 3] uint8; min -> blue (0,0,255), max -> red."""
    array = np.array([[0, 255], [0, 255]], dtype=np.uint8)
    path = _save_gray_png(tmp_path, "depth.png", array)
    rgb = _map_depth_image_to_rgb(depth_image_path=path)
    assert isinstance(rgb, torch.Tensor)
    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == torch.uint8
    # After min-subtraction, value 0 normalizes to 0 (palette start = blue),
    # value 255 normalizes to 1 (palette end = red).
    assert tuple(int(v) for v in rgb[0, 0]) == (0, 0, 255)
    assert tuple(int(v) for v in rgb[0, 1]) == (255, 0, 0)


def test_map_segmentation_image_to_rgb_shape_dtype_and_colors(tmp_path):
    """Segmentation helper yields [H, W, 3] uint8 with per-class palette colors."""
    array = np.array([[0, 1], [3, 0]], dtype=np.uint8)
    path = _save_gray_png(tmp_path, "seg.png", array)
    class_id_to_rgb = {
        0: get_class_color(class_id=0),
        1: get_class_color(class_id=1),
        3: get_class_color(class_id=3),
    }
    rgb = _map_segmentation_image_to_rgb(
        segmentation_image_path=path, class_id_to_rgb=class_id_to_rgb
    )
    assert isinstance(rgb, torch.Tensor)
    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == torch.uint8
    assert tuple(int(v) for v in rgb[0, 0]) == get_class_color(class_id=0)
    assert tuple(int(v) for v in rgb[0, 1]) == get_class_color(class_id=1)
    assert tuple(int(v) for v in rgb[1, 0]) == get_class_color(class_id=3)
