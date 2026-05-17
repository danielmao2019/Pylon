"""Scene-graph display response schema."""

from typing import Literal, Optional

from data.viewer.utils.atomic_displays.utils.ts.backend.schemas.display_response import (
    DisplayResponse,
)


class SceneGraphDisplayResponse(DisplayResponse):
    """Scene-graph display response.

    Args:
        None.

    Returns:
        Pydantic model for scene-graph displays.
    """

    display_kind: Literal["scene_graph"] = "scene_graph"
    original_overlay_url: Optional[str] = None
