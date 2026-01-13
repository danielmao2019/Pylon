from typing import Any, List, Optional, Tuple

import dash
from dash import MATCH
from dash.dependencies import Input, Output, State

from models.three_d.octree_gs.callbacks.common import (
    build_octree_main_figure,
    decode_trigger_id,
)


def register_octree_rgb_change(app: dash.Dash, viewer: Any) -> None:
    @app.callback(
        Output(
            {
                "type": "octree-main-image",
                "modality": "rgb",
                "dataset": MATCH,
                "scene": MATCH,
                "method": MATCH,
            },
            "figure",
        ),
        Output(
            {
                "type": "model-store",
                "dataset": MATCH,
                "scene": MATCH,
                "method": MATCH,
                "field": "octree_selected_levels_rgb",
            },
            "data",
            allow_duplicate=True,
        ),
        Input(
            {
                "type": "octree-levels-checklist-rgb",
                "dataset": MATCH,
                "scene": MATCH,
                "method": MATCH,
            },
            "value",
        ),
        State("camera-overlay-toggle-store", "data"),
        State(
            {
                "type": "model-store",
                "dataset": MATCH,
                "scene": MATCH,
                "method": MATCH,
                "field": "octree_selected_levels_rgb",
            },
            "data",
        ),
        prevent_initial_call=True,
    )
    def _on_octree_rgb_change(
        rgb_levels_store: Optional[List[int]],
        show_cameras_state: bool,
        _store_values: Optional[Any],
    ) -> Tuple[Any, Any]:
        assert rgb_levels_store is None or isinstance(
            rgb_levels_store, list
        ), f"{type(rgb_levels_store)=}"
        assert isinstance(
            show_cameras_state, bool
        ), "overlay toggle store data must be a boolean"
        triggered_id = decode_trigger_id()
        dataset_name = triggered_id["dataset"]
        scene_name = triggered_id["scene"]
        method_name = triggered_id["method"]
        figure = build_octree_main_figure(
            viewer=viewer,
            dataset_name=dataset_name,
            scene_name=scene_name,
            method_name=method_name,
            modality="rgb",
            selected_levels=rgb_levels_store,
            show_cameras=show_cameras_state,
        )
        return figure, rgb_levels_store
