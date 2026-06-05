"""Layered display response schema."""

from typing import List, Literal

from data.viewer.utils.atomic_displays.utils.ts.backend.schemas.display_response import (
    DisplayResponse,
)

RASTER_DISPLAY_KINDS = frozenset(
    {
        "color_image",
        "depth_image",
        "edge_image",
        "normal_image",
        "segmentation_image",
        "instance_surrogate_image",
        "video",
    }
)

SPATIAL_DISPLAY_KINDS = frozenset(
    {
        "color_pc",
        "segmentation_pc",
        "color_gs",
        "segmentation_gs",
        "scene_graph",
        "camera",
        # Meshes are spatial 3D displays, like point clouds and gaussians.
        "color_mesh",
        "segmentation_mesh",
        "heatmap_mesh",
        "sparse_heatmap_mesh",
    }
)


class LayeredDisplayResponse(DisplayResponse):
    """Layered display response.

    Args:
        None.

    Returns:
        Pydantic model carrying one base display response plus an ordered list
        of auxiliary display responses stacked on top of the base. Consumers
        own per-layer semantics and visibility state.
    """

    display_kind: Literal["layered"] = "layered"
    base_display_response: DisplayResponse
    aux_display_responses: List[DisplayResponse]

    def _display_class_of(self, layer: DisplayResponse) -> str:
        """Map a layer's display_kind to its composable display class.

        Args:
            layer: A single layer display response whose display_kind decides
                its composable display class.

        Returns:
            "placeholder" for the passive stand-in kind (compatible with any
            class), "raster" for kinds in RASTER_DISPLAY_KINDS, and "spatial"
            for kinds in SPATIAL_DISPLAY_KINDS. Raises ValueError for text,
            table, and other non-layerable kinds.
        """
        if layer.display_kind == "placeholder":
            return "placeholder"
        elif layer.display_kind in RASTER_DISPLAY_KINDS:
            return "raster"
        elif layer.display_kind in SPATIAL_DISPLAY_KINDS:
            return "spatial"
        else:
            raise ValueError(
                "Non-layerable display_kind cannot participate in a layered "
                f"display: display_kind={layer.display_kind!r}, "
                f"raster_kinds={sorted(RASTER_DISPLAY_KINDS)}, "
                f"spatial_kinds={sorted(SPATIAL_DISPLAY_KINDS)}."
            )

    def model_post_init(self, __context) -> None:
        """Reject a layered response whose layers do not share one display class.

        Args:
            __context: Pydantic post-construction context object (unused).

        Returns:
            None. Raises ValueError when the non-placeholder layers do not all
            resolve to a single composable display class.
        """
        layers = [self.base_display_response] + self.aux_display_responses
        resolved_classes = [self._display_class_of(layer) for layer in layers]
        non_placeholder_classes = [
            display_class
            for display_class in resolved_classes
            if display_class != "placeholder"
        ]
        if len(set(non_placeholder_classes)) > 1:
            raise ValueError(
                "Layered display layers must all resolve to a single composable "
                f"display class: non_placeholder_classes={non_placeholder_classes}, "
                f"resolved_classes={resolved_classes}."
            )
