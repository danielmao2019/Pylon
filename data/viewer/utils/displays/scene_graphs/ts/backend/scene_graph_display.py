"""Scene-graph display response API."""

from typing import Optional
from urllib.parse import urlencode

from data.viewer.utils.displays.scene_graphs.ts.backend.schemas.display_response import (
    SceneGraphDisplayResponse,
)


def create_scene_graph_display_response(
    slot_id: str,
    title: str,
    scene_graph_path: Optional[str],
    original_overlay_path: Optional[str],
) -> SceneGraphDisplayResponse:
    """Create a scene-graph display response.

    Args:
        slot_id: Stable display slot identifier.
        title: Display panel title.
        scene_graph_path: Scene-graph artifact path.
        original_overlay_path: Original scene artifact path.

    Returns:
        Scene-graph display response.
    """
    if scene_graph_path is None:
        url: Optional[str] = None
    else:
        url = "/api/artifacts?%s" % urlencode({"path": scene_graph_path})
    response = SceneGraphDisplayResponse(
        slot_id=slot_id,
        title=title,
        display_kind="scene_graph",
        url=url,
        meta_info={},
    )
    if original_overlay_path is not None:
        response.original_overlay_url = "/api/artifacts?%s" % urlencode(
            {"path": original_overlay_path},
        )
    return response
