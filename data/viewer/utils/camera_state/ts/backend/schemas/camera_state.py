"""TypeScript backend camera-state schema."""

from typing import Any, Dict, Optional

from pydantic import BaseModel


class CameraState(BaseModel):
    """Serialized spatial viewer camera state.

    Args:
        None.

    Returns:
        Pydantic model mirroring the frontend CameraState interface.
    """

    intrinsics: Dict[str, Any]
    extrinsics: Dict[str, Any]
    convention: str
    name: Optional[str] = None
    id: Optional[str] = None
