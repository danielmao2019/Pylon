"""Mesh display response core."""

from typing import Any, Dict, Optional, Type
from urllib.parse import urlencode

from data.viewer.utils.atomic_displays.mesh.ts.backend.schemas.display_response import (
    MeshDisplayResponse,
)


def create_mesh_display_response(
    response_type: Type[MeshDisplayResponse],
    slot_id: str,
    title: str,
    display_kind: str,
    mesh_path: Optional[str],
    meta_info: Dict[str, Any] | None = None,
) -> MeshDisplayResponse:
    """Create a mesh display response.

    Args:
        response_type: Concrete mesh display response model.
        slot_id: Stable display slot identifier.
        title: Display panel title.
        display_kind: Atomic display kind.
        mesh_path: Mesh artifact path.
        meta_info: Optional renderer metadata.

    Returns:
        Mesh display response.
    """
    assert issubclass(
        response_type, MeshDisplayResponse
    ), "Response type must be a MeshDisplayResponse subclass. response_type=%r" % (
        response_type,
    )
    assert mesh_path is None or isinstance(mesh_path, str), (
        "Mesh path must be None or a string. mesh_path=%r" % mesh_path
    )
    if mesh_path is None:
        url: Optional[str] = None
    else:
        url = "/api/artifacts?%s" % urlencode({"path": mesh_path})
    return response_type(
        slot_id=slot_id,
        title=title,
        display_kind=display_kind,
        url=url,
        meta_info={} if meta_info is None else dict(meta_info),
    )
