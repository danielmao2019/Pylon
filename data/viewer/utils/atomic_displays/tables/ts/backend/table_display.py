"""Table display response API."""

from typing import Optional
from urllib.parse import urlencode

from data.viewer.utils.atomic_displays.tables.ts.backend.schemas.display_response import (
    TableDisplayResponse,
)


def create_table_display_response(
    slot_id: str,
    title: str,
    table_path: Optional[str],
) -> TableDisplayResponse:
    """Create a table display response.

    Args:
        slot_id: Stable display slot identifier.
        title: Display panel title.
        table_path: Optional table artifact path.

    Returns:
        Table display response.
    """
    assert table_path is None or isinstance(table_path, str), (
        "Table path must be None or a string. table_path=%r" % table_path
    )
    if table_path is None:
        url: Optional[str] = None
    else:
        url = "/api/artifacts?%s" % urlencode({"path": table_path})
    return TableDisplayResponse(
        slot_id=slot_id,
        title=title,
        display_kind="table",
        url=url,
        meta_info={},
    )
