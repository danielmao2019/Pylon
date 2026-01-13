from typing import Any, List, Optional, Tuple

import dash
from dash import MATCH
from dash.dependencies import Input, Output, State

from models.three_d.lapis_gs.callbacks.common import (
    build_lapis_main_figure,
    decode_trigger_id,
    sanitize_layers,
    total_layers,
)


def register_lapis_density_change(app: dash.Dash, viewer: Any) -> None:
    @app.callback(
        Output(
            {
                'type': 'lapis-main-image',
                'modality': 'density',
                'dataset': MATCH,
                'scene': MATCH,
                'method': MATCH,
            },
            'figure',
        ),
        Output(
            {
                'type': 'model-store',
                'dataset': MATCH,
                'scene': MATCH,
                'method': MATCH,
                'field': 'lapis_selected_layers_density',
            },
            'data',
            allow_duplicate=True,
        ),
        Input(
            {
                'type': 'lapis-layers-checklist-density',
                'dataset': MATCH,
                'scene': MATCH,
                'method': MATCH,
            },
            'value',
        ),
        State("camera-overlay-toggle-store", "data"),
        State(
            {
                'type': 'model-store',
                'dataset': MATCH,
                'scene': MATCH,
                'method': MATCH,
                'field': 'lapis_selected_layers_density',
            },
            'data',
        ),
        prevent_initial_call=True,
    )
    def _on_lapis_density_change(
        selected_values: Optional[List[int]],
        show_cameras_state: bool,
        store_values: Optional[Any],
    ) -> Tuple[Any, List[int]]:
        assert isinstance(
            show_cameras_state, bool
        ), "overlay toggle store data must be a boolean"
        triggered_id = decode_trigger_id()
        dataset_name = triggered_id["dataset"]
        scene_name = triggered_id["scene"]
        method_name = triggered_id["method"]
        total = total_layers(viewer, dataset_name, scene_name, method_name)
        layers = sanitize_layers(selected_values, store_values, total)
        figure = build_lapis_main_figure(
            viewer=viewer,
            dataset_name=dataset_name,
            scene_name=scene_name,
            method_name=method_name,
            modality='density',
            selected_layers=layers,
            show_cameras=show_cameras_state,
        )
        return figure, layers
