"""Shared context for the dataset viewer Dash app."""

from typing import Dict, Optional

from data.viewer.backend import ViewerBackend

_VIEWER_CONTEXT: "DatasetViewerContext | None" = None


class DatasetViewerContext:
    def __init__(
        self, backend: ViewerBackend, available_datasets: Dict[str, Dict[str, str]]
    ) -> None:
        # Input validations
        assert isinstance(
            backend, ViewerBackend
        ), f"backend must be ViewerBackend, got {type(backend)}"
        assert isinstance(
            available_datasets, dict
        ), f"available_datasets must be dict, got {type(available_datasets)}"

        self.backend = backend
        self.available_datasets = available_datasets


def set_viewer_context(context: DatasetViewerContext) -> None:
    # Input validations
    assert isinstance(
        context, DatasetViewerContext
    ), f"context must be DatasetViewerContext, got {type(context)}"

    global _VIEWER_CONTEXT
    _VIEWER_CONTEXT = context


def get_viewer_context() -> DatasetViewerContext:
    assert (
        _VIEWER_CONTEXT is not None
    ), "Dataset viewer context has not been initialized"
    return _VIEWER_CONTEXT
