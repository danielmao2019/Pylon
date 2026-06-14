"""Base atomic display response schema."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class DisplayResponse(BaseModel):
    """Base atomic display response.

    Args:
        None.

    Returns:
        Pydantic model carrying renderer instructions for a single display.
    """

    slot_id: str
    title: str
    display_kind: str
    url: Optional[str] = None
    original_overlay_url: Optional[str] = None
    meta_info: Dict[str, Any] = Field(default_factory=dict)
