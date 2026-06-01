"""Core Dash pixel display API."""

from typing import Any

import numpy as np
import plotly.graph_objects as go
import torch
from dash import dcc

from data.viewer.utils.atomic_displays.pixels.dash.image_display import (
    create_image_display,
    get_image_display_stats,
)
from data.viewer.utils.atomic_displays.pixels.dash.segmentation_display import (
    create_segmentation_display,
    get_segmentation_display_stats,
)

__all__ = [
    "create_dash_pixels_display",
    "create_image_display",
    "create_segmentation_display",
    "get_image_display_stats",
    "get_segmentation_display_stats",
]


def create_dash_pixels_display(image: Any, image_interpolation: str) -> dcc.Graph:
    """Render a resolved RGB pixel image as a Dash graph; modality-agnostic.

    Args:
        image: Resolved RGB image of shape [H, W, 3] as a uint8 ``numpy.ndarray``
            or ``torch.Tensor`` with channel values in ``[0, 255]``.
        image_interpolation: Interpolation choice applied to the Plotly figure.
            ``nearest`` disables smoothing (``zsmooth=False``); any other value
            enables smoothing.

    Returns:
        Dash ``dcc.Graph`` rendering the RGB image.
    """
    assert isinstance(
        image, (np.ndarray, torch.Tensor)
    ), "Image must be a numpy.ndarray or torch.Tensor. type(image)=%r" % (type(image),)
    assert isinstance(
        image_interpolation, str
    ), "Image interpolation must be a string. image_interpolation=%r" % (
        image_interpolation,
    )

    if isinstance(image, torch.Tensor):
        image_array = image.detach().cpu().numpy()
    else:
        image_array = image
    assert (
        image_array.ndim == 3 and image_array.shape[2] == 3
    ), "Image must have shape [H, W, 3]. image_array.shape=%r" % (image_array.shape,)

    zsmooth: Any = False if image_interpolation == "nearest" else "fast"
    figure = go.Figure(data=go.Image(z=image_array, zsmooth=zsmooth))
    figure.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return dcc.Graph(figure=figure)
