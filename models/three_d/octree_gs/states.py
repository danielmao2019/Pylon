from typing import Any, Type

import dash
from dash import dcc

from data.viewer.ivision.layout import MODEL_STORE_CONTAINER_ID


def _get_model_store_container(app: dash.Dash) -> Any:
    layout = app.layout
    assert hasattr(layout, "children"), "app layout must expose children"
    layout_children = layout.children
    assert isinstance(layout_children, list), "app layout children must be a list"
    assert (
        len(layout_children) >= 2
    ), "app layout must include model store container as second child"
    container = layout_children[1]
    assert getattr(container, "id", None) == MODEL_STORE_CONTAINER_ID
    return container


def setup_states(
    app: dash.Dash,
    scene_model_cls: Type[Any],
    dataset_name: str,
    scene_name: str,
    method_name: str,
) -> None:
    """Initialize default modality state entries for Octree GS scenes."""
    assert scene_model_cls is not None
    container = _get_model_store_container(app)

    for field, default in (
        ('octree_selected_levels_rgb', None),
        ('octree_selected_levels_density', None),
        ('octree_debugger_enabled', False),
    ):
        children = container.children
        assert isinstance(children, list), "model store container children must be list"
        children.append(
            dcc.Store(
                id={
                    'type': 'model-store',
                    'dataset': dataset_name,
                    'scene': scene_name,
                    'method': method_name,
                    'field': field,
                },
                data=default,
            )
        )
