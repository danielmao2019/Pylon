"""Dash Gaussian-splat display APIs.

These APIs match the skeleton declaration for the Dash Gaussian display
modality. They are not exercised by any caller in this branch; the benchmark
viewer renders Gaussians through `gaussians/ts/backend/apis.py`.
"""

from typing import Any, Dict, Optional

import torch  # noqa: F401 - skeleton import for the Dash segmentation Gaussian impl

from data.viewer.utils.atomic_displays.gaussians.dash.core_gaussians_display import (
    create_dash_gaussians_display,
)
from data.viewer.utils.atomic_displays.utils.class_colors import (  # noqa: F401
    map_class_ids_to_rgb,
)


def create_color_gs_display(
    gaussian_path: Optional[str],
    title: str,
    meta_info: Optional[Dict[str, Any]] = None,
) -> Any:
    """Create a Dash Gaussian display for a color Gaussian artifact.

    Args:
        gaussian_path: Color Gaussian artifact path.
        title: Display panel title.
        meta_info: Optional renderer metadata.

    Returns:
        Dash Gaussian-splat display component.
    """
    assert gaussian_path is None or isinstance(gaussian_path, str), (
        "Gaussian path must be None or a string. gaussian_path=%r" % gaussian_path
    )
    assert isinstance(title, str), "Title must be a string. title=%r" % title
    assert meta_info is None or isinstance(meta_info, dict), (
        "Meta info must be None or a dict. meta_info=%r" % meta_info
    )
    return create_dash_gaussians_display(
        gaussian_path=gaussian_path,
        title=title,
        meta_info=meta_info,
    )


def create_segmentation_gs_display(
    segmentation_gs_path: Optional[str],
    title: str,
    meta_info: Optional[Dict[str, Any]] = None,
) -> Any:
    """Create a Dash Gaussian display for a segmentation Gaussian artifact.

    Args:
        segmentation_gs_path: Segmentation Gaussian artifact path.
        title: Display panel title.
        meta_info: Optional renderer metadata.

    Returns:
        Dash Gaussian-splat display component.
    """
    assert segmentation_gs_path is None or isinstance(segmentation_gs_path, str), (
        "Segmentation Gaussian path must be None or a string. "
        "segmentation_gs_path=%r" % segmentation_gs_path
    )
    assert isinstance(title, str), "Title must be a string. title=%r" % title
    assert meta_info is None or isinstance(meta_info, dict), (
        "Meta info must be None or a dict. meta_info=%r" % meta_info
    )
    if segmentation_gs_path is None:
        return create_dash_gaussians_display(
            gaussian_path=None,
            title=title,
            meta_info=meta_info,
        )
    raise NotImplementedError(
        "Dash segmentation Gaussian rendering is declared by the skeleton but "
        "not exercised by any caller in this branch. The benchmark viewer "
        "renders segmentation Gaussians through "
        "data.viewer.utils.atomic_displays.gaussians.ts.backend.apis."
    )


def _map_segmentation_gs_to_rgb(
    segmentation_gs_path: str,
    class_id_to_rgb: Dict[int, Any],
) -> str:
    """Render a colored Gaussian artifact from class-labeled Gaussians.

    Args:
        segmentation_gs_path: Segmentation Gaussian artifact path.
        class_id_to_rgb: Class-id to RGB mapping.

    Returns:
        Path to a color Gaussian artifact whose RGB matches `class_id_to_rgb`.
    """
    assert isinstance(segmentation_gs_path, str), (
        "Segmentation Gaussian path must be a string. "
        "segmentation_gs_path=%r" % segmentation_gs_path
    )
    assert isinstance(
        class_id_to_rgb, dict
    ), "Class id to RGB must be a dict. class_id_to_rgb=%r" % (class_id_to_rgb,)
    raise NotImplementedError(
        "Dash segmentation-to-color Gaussian mapping is declared by the "
        "skeleton but not exercised by any caller in this branch."
    )
