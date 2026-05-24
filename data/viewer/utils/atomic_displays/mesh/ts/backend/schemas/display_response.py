"""Mesh display response schemas."""

from typing import Literal

from data.viewer.utils.atomic_displays.utils.ts.backend.schemas.display_response import (
    DisplayResponse,
)


class MeshDisplayResponse(DisplayResponse):
    """Base mesh display response.

    Concrete leaf subclasses (``ColorMeshDisplayResponse``,
    ``HeatmapMeshDisplayResponse``, ``SegmentationMeshDisplayResponse``)
    override ``display_kind`` with their kind-specific ``Literal``. The
    default value here is only used by the internal helper
    :func:`create_mesh_display_response`, whose return value is consumed
    only by leaf factories that rebuild the concrete response.

    Args:
        None.

    Returns:
        Pydantic model for mesh display responses.
    """

    display_kind: str = "mesh"


class ColorMeshDisplayResponse(MeshDisplayResponse):
    """Color mesh display response.

    Args:
        None.

    Returns:
        Pydantic model for color mesh displays.
    """

    display_kind: Literal["color_mesh"] = "color_mesh"


class SegmentationMeshDisplayResponse(MeshDisplayResponse):
    """Segmentation mesh display response.

    Args:
        None.

    Returns:
        Pydantic model for segmentation mesh displays.
    """

    display_kind: Literal["segmentation_mesh"] = "segmentation_mesh"


class HeatmapMeshDisplayResponse(MeshDisplayResponse):
    """Heatmap mesh display response.

    Args:
        None.

    Returns:
        Pydantic model for heatmap mesh displays.
    """

    display_kind: Literal["heatmap_mesh"] = "heatmap_mesh"


class SparseHeatmapMeshDisplayResponse(MeshDisplayResponse):
    """Sparse heatmap mesh display response.

    Carries only the per-vertex (indices, values) delta as the wire resource;
    rendered by overlaying those values onto the surrounding
    ``LayeredDisplayResponse.base_display_response`` mesh at render time.

    Args:
        None.

    Returns:
        Pydantic model for sparse heatmap mesh displays.
    """

    display_kind: Literal["sparse_heatmap_mesh"] = "sparse_heatmap_mesh"
