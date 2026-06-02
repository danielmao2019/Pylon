"""Tests for layered display response modality type-compatibility check.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""

import pytest

from data.viewer.utils.atomic_displays.utils.ts.backend.schemas.display_response import (
    DisplayResponse,
)
from data.viewer.utils.atomic_displays.utils.ts.backend.schemas.layered_display_response import (
    LayeredDisplayResponse,
)


def _make_layer(display_kind: str) -> DisplayResponse:
    """Build a minimal base DisplayResponse with the given display_kind.

    Args:
        display_kind: The display_kind value driving the compatibility class.

    Returns:
        A minimally-populated DisplayResponse carrying the given display_kind.
    """
    return DisplayResponse(slot_id="slot", title="title", display_kind=display_kind)


def test_all_raster_layers_constructs():
    """All-raster layers (color_image base + depth_image aux) construct OK."""
    response = LayeredDisplayResponse(
        slot_id="slot",
        title="title",
        base_display_response=_make_layer("color_image"),
        aux_display_responses=[_make_layer("depth_image")],
    )
    assert response.display_kind == "layered"


def test_all_spatial_layers_constructs():
    """All-spatial layers (color_pc base + camera aux) construct OK."""
    response = LayeredDisplayResponse(
        slot_id="slot",
        title="title",
        base_display_response=_make_layer("color_pc"),
        aux_display_responses=[_make_layer("camera")],
    )
    assert response.display_kind == "layered"


def test_placeholder_mixed_with_raster_constructs():
    """A placeholder layer does not constrain a raster resolved class."""
    response = LayeredDisplayResponse(
        slot_id="slot",
        title="title",
        base_display_response=_make_layer("color_image"),
        aux_display_responses=[_make_layer("placeholder")],
    )
    assert response.display_kind == "layered"


def test_placeholder_mixed_with_spatial_constructs():
    """A placeholder layer does not constrain a spatial resolved class."""
    response = LayeredDisplayResponse(
        slot_id="slot",
        title="title",
        base_display_response=_make_layer("placeholder"),
        aux_display_responses=[_make_layer("color_pc")],
    )
    assert response.display_kind == "layered"


def test_all_placeholder_constructs():
    """All-placeholder layers construct OK (no non-placeholder class to clash)."""
    response = LayeredDisplayResponse(
        slot_id="slot",
        title="title",
        base_display_response=_make_layer("placeholder"),
        aux_display_responses=[_make_layer("placeholder")],
    )
    assert response.display_kind == "layered"


def test_raster_base_spatial_aux_raises():
    """Mixing a raster base with a spatial aux raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        LayeredDisplayResponse(
            slot_id="slot",
            title="title",
            base_display_response=_make_layer("color_image"),
            aux_display_responses=[_make_layer("color_pc")],
        )
    assert "single composable" in str(exc_info.value)


def test_text_layer_raises():
    """A non-layerable text kind as a layer raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        LayeredDisplayResponse(
            slot_id="slot",
            title="title",
            base_display_response=_make_layer("text"),
            aux_display_responses=[_make_layer("color_image")],
        )
    assert "Non-layerable display_kind" in str(exc_info.value)


def test_table_layer_raises():
    """A non-layerable table kind as a layer raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        LayeredDisplayResponse(
            slot_id="slot",
            title="title",
            base_display_response=_make_layer("color_image"),
            aux_display_responses=[_make_layer("table")],
        )
    assert "Non-layerable display_kind" in str(exc_info.value)
