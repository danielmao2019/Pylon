"""Gaussian-splat display response core."""

from typing import Any, Dict, Optional, Type
from urllib.parse import urlencode

from data.viewer.utils.atomic_displays.gaussians.ts.backend.schemas.display_response import (
    GaussianDisplayResponse,
)


def create_gaussians_display_response_core(
    response_type: Type[GaussianDisplayResponse],
    slot_id: str,
    title: str,
    display_kind: str,
    gaussian_path: Optional[str],
    meta_info: Dict[str, Any] | None = None,
) -> GaussianDisplayResponse:
    """Create a Gaussian-splat display response.

    Args:
        response_type: Concrete Gaussian display response model.
        slot_id: Stable display slot identifier.
        title: Display panel title.
        display_kind: Atomic display kind.
        gaussian_path: Gaussian-splat artifact path.
        meta_info: Optional renderer metadata.

    Returns:
        Gaussian-splat display response.
    """
    assert issubclass(
        response_type, GaussianDisplayResponse
    ), "Response type must be a GaussianDisplayResponse subclass. response_type=%r" % (
        response_type,
    )
    assert isinstance(slot_id, str), "Slot id must be a string. slot_id=%r" % slot_id
    assert isinstance(title, str), "Title must be a string. title=%r" % title
    assert isinstance(display_kind, str), (
        "Display kind must be a string. display_kind=%r" % display_kind
    )
    assert gaussian_path is None or isinstance(gaussian_path, str), (
        "Gaussian path must be None or a string. gaussian_path=%r" % gaussian_path
    )
    assert meta_info is None or isinstance(meta_info, dict), (
        "Meta info must be None or a dict. meta_info=%r" % meta_info
    )

    if gaussian_path is None:
        url: Optional[str] = None
    else:
        url = "/api/artifacts?%s" % urlencode({"path": gaussian_path})

    return response_type(
        slot_id=slot_id,
        title=title,
        display_kind=display_kind,
        url=url,
        meta_info={} if meta_info is None else dict(meta_info),
    )
