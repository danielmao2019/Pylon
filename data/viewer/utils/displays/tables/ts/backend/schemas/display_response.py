"""Table display response schema."""

from typing import Literal

from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import (
    DisplayResponse,
)


class TableDisplayResponse(DisplayResponse):
    """Table display response.

    Args:
        None.

    Returns:
        Pydantic model for table displays.
    """

    display_kind: Literal["table"] = "table"
