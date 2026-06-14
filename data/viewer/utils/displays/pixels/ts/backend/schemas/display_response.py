"""Pixel display response schemas."""

from typing import Literal, Optional

from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import (
    DisplayResponse,
)


class PixelDisplayResponse(DisplayResponse):
    """Base pixel display response.

    Args:
        None.

    Returns:
        Pydantic model for pixel display responses.
    """


class ColorImageDisplayResponse(PixelDisplayResponse):
    """Color image display response.

    Args:
        None.

    Returns:
        Pydantic model for color image displays.
    """

    display_kind: Literal["color_image"] = "color_image"


class DepthImageDisplayResponse(PixelDisplayResponse):
    """Depth image display response.

    Args:
        None.

    Returns:
        Pydantic model for depth image displays.
    """

    display_kind: Literal["depth_image"] = "depth_image"


class EdgeImageDisplayResponse(PixelDisplayResponse):
    """Edge image display response.

    Args:
        None.

    Returns:
        Pydantic model for edge image displays.
    """

    display_kind: Literal["edge_image"] = "edge_image"


class NormalImageDisplayResponse(PixelDisplayResponse):
    """Normal image display response.

    Args:
        None.

    Returns:
        Pydantic model for normal image displays.
    """

    display_kind: Literal["normal_image"] = "normal_image"


class SegmentationImageDisplayResponse(PixelDisplayResponse):
    """Segmentation image display response.

    Args:
        None.

    Returns:
        Pydantic model for segmentation image displays.
    """

    display_kind: Literal["segmentation_image"] = "segmentation_image"
    original_overlay_url: Optional[str] = None


class InstanceSurrogateImageDisplayResponse(PixelDisplayResponse):
    """Instance-surrogate image display response.

    Args:
        None.

    Returns:
        Pydantic model for instance-surrogate image displays.
    """

    display_kind: Literal["instance_surrogate_image"] = "instance_surrogate_image"
    original_overlay_url: Optional[str] = None
