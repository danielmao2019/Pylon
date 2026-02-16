"""Dash layout builder for the 3D scene viewer."""

from typing import Any, Dict, List, Optional

from dash import Dash, dcc, html
from dash_extensions import Keyboard

from data.viewer.three_d_scene.layout.styles import (
    CAMERA_INFO_CONTAINER_STYLE,
    CAMERA_INFO_HEADER_STYLE,
    CAMERA_INFO_TEXT_STYLE,
    CAMERA_OVERLAY_CONTAINER_STYLE,
    CAMERA_SELECTOR_CONTAINER_STYLE,
    CAMERA_SELECTOR_HEADER_STYLE,
    CAMERA_SELECTOR_INPUT_STYLE,
    CAMERA_SELECTOR_LABEL_STYLE,
    CAMERA_SELECTOR_SCROLLABLE_STYLE,
    CAMERA_SELECTOR_SPLIT_CONTAINER_STYLE,
    CAMERA_SELECTOR_SPLIT_TITLE_STYLE,
    CONTROL_HEADER_STYLE,
    KEYBOARD_SHORTCUTS_HEADER_STYLE,
    KEYBOARD_SHORTCUTS_LIST_STYLE,
    KEYBOARD_STYLE,
    LABEL_BOLD_STYLE,
    LAYOUT_WRAPPER_STYLE,
    MAIN_PANEL_STYLE,
    MODEL_STORE_CONTAINER_STYLE,
    PAGE_STYLE,
    PRIMARY_BUTTON_STYLE,
    RECORD_STATUS_STYLE,
    SIDE_PANEL_STYLE,
    SLIDER_BLOCK_STYLE,
    SLIDER_COMPONENT_STYLE,
    SLIDER_LABEL_STYLE,
    SLIDER_ROW_STYLE,
    SLIDER_VALUE_STYLE,
    _make_grid_style,
)

CAMERA_NONE_VALUE = "__none__"
CAMERA_SELECTOR_RADIO_TYPE = "camera-radio"
CAMERA_SELECTOR_ROOT_ID = "camera-selector-root"
CAMERA_OVERLAY_TOGGLE_STORE_ID = "camera-overlay-toggle-store"
CAMERA_OVERLAY_TOGGLE_BUTTON_ID = "camera-overlay-toggle-button"
MODEL_STORE_CONTAINER_ID = "model-store-container"


def build_layout(app: Dash, dataset_options: List[Dict[str, Any]]) -> None:
    # Input validations
    assert isinstance(app, Dash), f"app must be Dash, got {type(app)}"
    assert isinstance(
        dataset_options, list
    ), f"dataset_options must be list, got {type(dataset_options)}"
    assert all(
        isinstance(option, dict) for option in dataset_options
    ), "dataset_options entries must be dicts"

    layout = build_app_layout(dataset_options=dataset_options)
    app.layout = layout


def build_app_layout(dataset_options: List[Dict[str, Any]]) -> html.Div:
    side_panel = _build_side_panel_layout(
        dataset_options=dataset_options,
        initial_scene_options=[],
        initial_scene_selection=None,
        initial_dataset_selection=None,
        initial_camera_info=None,
    )
    main_panel = _build_main_panel_layout(
        initial_grid_children=[],
        initial_grid_style=_make_grid_style(0),
    )
    model_store_container = html.Div(
        [],
        id=MODEL_STORE_CONTAINER_ID,
        style=MODEL_STORE_CONTAINER_STYLE,
    )
    layout_wrapper = html.Div(
        [side_panel, main_panel],
        style={**LAYOUT_WRAPPER_STYLE},
    )
    return html.Div([layout_wrapper, model_store_container], style={**PAGE_STYLE})


# ---------------------------
# Side panel
# ---------------------------


def _build_side_panel_layout(
    dataset_options: List[Dict[str, Any]],
    initial_scene_options: List[Dict[str, Any]],
    initial_scene_selection: Optional[str],
    initial_dataset_selection: Optional[str],
    initial_camera_info: Optional[Dict[str, Any]],
) -> html.Div:
    dataset_selector = _build_dataset_dropdown_layout(
        dataset_options=dataset_options,
        initial_dataset_selection=initial_dataset_selection,
    )
    scene_selector = _build_scene_dropdown_layout(
        initial_scene_options=initial_scene_options,
        initial_scene_selection=initial_scene_selection,
    )
    controls_block = _build_controls_layout()
    camera_info_block = _build_camera_info_layout(initial_camera_info)
    camera_selector_block = _build_camera_selector_container_layout(
        initial_dataset_selection=initial_dataset_selection,
        initial_scene_selection=initial_scene_selection,
    )
    camera_overlay_toggle = _build_camera_overlay_toggle_layout()
    keyboard_shortcuts = _build_keyboard_shortcuts_layout()
    return html.Div(
        [
            dataset_selector,
            scene_selector,
            controls_block,
            camera_info_block,
            camera_selector_block,
            camera_overlay_toggle,
            keyboard_shortcuts,
        ],
        style={**SIDE_PANEL_STYLE},
    )


def _build_dataset_dropdown_layout(
    dataset_options: List[Dict[str, Any]], initial_dataset_selection: Optional[str]
) -> html.Div:
    return html.Div(
        [
            html.Label("Dataset", style=LABEL_BOLD_STYLE),
            dcc.Dropdown(
                id="dataset-dropdown",
                options=dataset_options,
                value=initial_dataset_selection,
                clearable=True,
            ),
            html.Br(),
        ]
    )


def _build_scene_dropdown_layout(
    initial_scene_options: List[Dict[str, Any]], initial_scene_selection: Optional[str]
) -> html.Div:
    return html.Div(
        [
            html.Label("Scene", style=LABEL_BOLD_STYLE),
            dcc.Dropdown(
                id="scene-dropdown",
                options=initial_scene_options,
                value=initial_scene_selection,
                clearable=True,
            ),
            html.Br(),
        ]
    )


def _build_controls_layout() -> html.Div:
    return html.Div(
        [
            html.H4("Controls", style=CONTROL_HEADER_STYLE),
            _build_translation_slider_layout(),
            _build_rotation_slider_layout(),
            _build_record_controls_layout(),
        ]
    )


def _build_translation_slider_layout() -> html.Div:
    slider_props = _make_slider_props()
    return html.Div(
        [
            html.Label("Translation Step:", style=SLIDER_LABEL_STYLE),
            html.Div(
                [
                    html.Div(
                        dcc.Slider(
                            id="translation-step-slider",
                            **slider_props,
                        ),
                        style=SLIDER_COMPONENT_STYLE,
                    ),
                    html.Span(
                        id="translation-step-display",
                        children="Step: --",
                        style=SLIDER_VALUE_STYLE,
                    ),
                ],
                style=SLIDER_ROW_STYLE,
            ),
        ],
        style=SLIDER_BLOCK_STYLE,
    )


def _build_rotation_slider_layout() -> html.Div:
    slider_props = _make_slider_props()
    return html.Div(
        [
            html.Label("Rotation Step:", style=SLIDER_LABEL_STYLE),
            html.Div(
                [
                    html.Div(
                        dcc.Slider(
                            id="rotation-step-slider",
                            **slider_props,
                        ),
                        style=SLIDER_COMPONENT_STYLE,
                    ),
                    html.Span(
                        id="rotation-step-display",
                        children="Step: --",
                        style=SLIDER_VALUE_STYLE,
                    ),
                ],
                style=SLIDER_ROW_STYLE,
            ),
        ],
        style=SLIDER_BLOCK_STYLE,
    )


def _build_record_controls_layout() -> html.Div:
    return html.Div(
        [
            html.Button(
                "Record Camera Extrinsics",
                id="record-button",
                n_clicks=0,
                style=PRIMARY_BUTTON_STYLE,
            ),
            html.Div(
                id="record-status",
                children="No cameras recorded yet",
                style=RECORD_STATUS_STYLE,
            ),
        ],
        style=SLIDER_BLOCK_STYLE,
    )


def _build_camera_info_layout(
    initial_camera_info: Optional[Dict[str, Any]],
) -> html.Div:
    text = format_camera_info_text(initial_camera_info)
    return html.Div(
        [
            html.H5("Camera Info:", style=CAMERA_INFO_HEADER_STYLE),
            html.Div(
                id="camera-info-display",
                children=text,
                style=CAMERA_INFO_TEXT_STYLE,
            ),
        ],
        style=CAMERA_INFO_CONTAINER_STYLE,
    )


def _build_camera_selector_container_layout(
    initial_dataset_selection: Optional[str], initial_scene_selection: Optional[str]
) -> html.Div:
    selector_component = _build_camera_selector_layout(
        splits={},
        choice=None,
        dataset=initial_dataset_selection or "",
        scene=initial_scene_selection or "",
    )
    return html.Div(
        selector_component,
        id=CAMERA_SELECTOR_ROOT_ID,
    )


def _build_camera_overlay_toggle_layout() -> html.Div:
    button_label = "Show Cameras"
    return html.Div(
        [
            dcc.Store(
                id=CAMERA_OVERLAY_TOGGLE_STORE_ID,
                data=False,
            ),
            html.Button(
                button_label,
                id=CAMERA_OVERLAY_TOGGLE_BUTTON_ID,
                n_clicks=0,
                style=PRIMARY_BUTTON_STYLE,
            ),
        ],
        style=CAMERA_OVERLAY_CONTAINER_STYLE,
    )


def _build_camera_selector_layout(
    splits: Dict[str, List[Any]],
    choice: Optional[Any],
    dataset: str,
    scene: str,
) -> html.Div:
    keys = list(splits.keys())
    has_partition = {"train", "val", "test"}.issubset(set(keys))
    if has_partition:
        ordered_keys = ["train", "val", "test"]
    elif not keys:
        ordered_keys = ["all"]
    elif "all" in keys:
        ordered_keys = ["all"] + [k for k in keys if k != "all"]
    else:
        ordered_keys = keys
    components: List[Any] = [
        html.H5("Camera Selection:", style=CAMERA_SELECTOR_HEADER_STYLE),
    ]

    common_radio_kwargs = {
        "inputStyle": CAMERA_SELECTOR_INPUT_STYLE,
        "labelStyle": CAMERA_SELECTOR_LABEL_STYLE,
    }

    for split_key in ordered_keys:
        entries = splits.get(split_key, [])
        if not entries:
            continue

        radio_value: Any = None
        if choice is not None:
            for option in entries:
                if option["value"] == choice:
                    radio_value = option["value"]
                    break
        else:
            radio_value = CAMERA_NONE_VALUE

        radio_entries = [{"label": "None", "value": CAMERA_NONE_VALUE}] + entries

        radio = dcc.RadioItems(
            id={
                "type": CAMERA_SELECTOR_RADIO_TYPE,
                "dataset": dataset,
                "scene": scene,
                "split": split_key,
            },
            options=radio_entries,
            value=radio_value,
            **common_radio_kwargs,
        )

        components.append(
            html.Div(
                [
                    html.H6(
                        split_key.title(),
                        style=CAMERA_SELECTOR_SPLIT_TITLE_STYLE,
                    ),
                    html.Div(radio, style=CAMERA_SELECTOR_SCROLLABLE_STYLE),
                ],
                style=CAMERA_SELECTOR_SPLIT_CONTAINER_STYLE,
            )
        )

    return html.Div(components, style=CAMERA_SELECTOR_CONTAINER_STYLE)


def _build_keyboard_shortcuts_layout() -> html.Div:
    return html.Div(
        [
            html.H5(
                "Keyboard Shortcuts:",
                style=KEYBOARD_SHORTCUTS_HEADER_STYLE,
            ),
            html.Ul(
                [
                    html.Li("W/S - Move forward/backward horizontally"),
                    html.Li("A/D - Move left/right"),
                    html.Li("F/R - Move along view ray"),
                    html.Li("Space/Shift - Move up/down"),
                    html.Li("Arrow keys - Rotate camera"),
                ],
                style=KEYBOARD_SHORTCUTS_LIST_STYLE,
            ),
        ]
    )


# ---------------------------
# Main panel
# ---------------------------


def _build_main_panel_layout(
    initial_grid_children: List[Any],
    initial_grid_style: Dict[str, Any],
) -> html.Div:
    grid_block = _build_image_grid_layout(
        initial_grid_children=initial_grid_children,
        initial_grid_style=initial_grid_style,
    )
    keyboard_component = _build_keyboard_overlay_layout()
    return html.Div(
        [
            grid_block,
            keyboard_component,
        ],
        style={**MAIN_PANEL_STYLE},
    )


def _build_image_grid_layout(
    initial_grid_children: List[Any], initial_grid_style: Dict[str, Any]
) -> html.Div:
    return html.Div(
        id="image-grid",
        children=initial_grid_children,
        style=initial_grid_style,
    )


def _build_keyboard_overlay_layout() -> Keyboard:
    return Keyboard(
        id="keyboard",
        captureKeys=[
            "w",
            "a",
            "s",
            "d",
            " ",
            "Shift",
            "f",
            "r",
            "ArrowUp",
            "ArrowDown",
            "ArrowLeft",
            "ArrowRight",
        ],
        style={**KEYBOARD_STYLE},
    )


# ---------------------------
# Shared helpers
# ---------------------------


def format_camera_info_text(info: Optional[Dict[str, Any]]) -> str:
    if not info:
        return ""

    frustum_width_px, frustum_height_px = info["frustum_resolution"]
    position = info["position"]
    lines = [
        "Intrinsics:",
        f"  FOVx: {info['fov_x']:.2f}°  FOVy: {info['fov_y']:.2f}°",
        f"  Frustum resolution: {frustum_width_px} x {frustum_height_px}",
        "",
        "Extrinsics:",
        f"  Position: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]",
        f"  Pitch: {info['pitch']:.1f}°  Yaw: {info['yaw']:.1f}°",
    ]
    return "\n".join(lines)


def _make_slider_props() -> Dict[str, Any]:
    marks: Dict[float, str] = {}
    tick_count = 20
    step = 1.0 / float(tick_count - 1)
    for idx in range(tick_count):
        position = round(min(idx * step, 1.0), 4)
        marks[position] = f"{position:.2f}"
    marks[1.0] = f"{1.0:.2f}"
    return {
        "min": 0.0,
        "max": 1.0,
        "step": None,
        "value": 0.1,
        "updatemode": "drag",
        "marks": marks,
        "tooltip": {
            "placement": "bottom",
            "always_visible": False,
        },
    }
