"""Shared context for the iVISION 4D scene viewer Dash app."""

from typing import Any

_VIEWER_CONTEXT: "IVisionViewerContext | None" = None


class IVisionViewerContext:
    def __init__(self, viewer: Any) -> None:
        # Input validations
        assert viewer is not None, "viewer must not be None"

        self.viewer = viewer


def set_viewer_context(context: IVisionViewerContext) -> None:
    # Input validations
    assert isinstance(
        context, IVisionViewerContext
    ), f"context must be IVisionViewerContext, got {type(context)}"

    global _VIEWER_CONTEXT
    _VIEWER_CONTEXT = context


def get_viewer_context() -> IVisionViewerContext:
    assert (
        _VIEWER_CONTEXT is not None
    ), "iVISION viewer context has not been initialized"
    return _VIEWER_CONTEXT
