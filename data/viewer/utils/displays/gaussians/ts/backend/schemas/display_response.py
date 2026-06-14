"""Gaussian-splat display response schemas."""

from typing import Literal, Optional

from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import (
    DisplayResponse,
)


class GaussianDisplayResponse(DisplayResponse):
    """Base Gaussian-splat display response.

    Args:
        None.

    Returns:
        Pydantic model for Gaussian-splat display responses.
    """


class ColorGSDisplayResponse(GaussianDisplayResponse):
    """Color Gaussian-splat display response.

    Args:
        None.

    Returns:
        Pydantic model for color Gaussian-splat displays.
    """

    display_kind: Literal["color_gs"] = "color_gs"


class SegmentationGSDisplayResponse(GaussianDisplayResponse):
    """Segmentation Gaussian-splat display response.

    Args:
        None.

    Returns:
        Pydantic model for segmentation Gaussian-splat displays.
    """

    display_kind: Literal["segmentation_gs"] = "segmentation_gs"
    original_overlay_url: Optional[str] = None
