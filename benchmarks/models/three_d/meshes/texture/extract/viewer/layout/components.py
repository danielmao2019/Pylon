"""Layout component builders for the texture-extraction benchmark viewer."""

from typing import Any, Dict, List

import torch
from dash import dcc, html

from benchmarks.models.three_d.meshes.texture.extract.viewer.backend.benchmark_backend import (
    build_scene_timing_figure,
)
from benchmarks.models.three_d.meshes.texture.extract.viewer.layout.styles import (
    CARD_STYLE,
    GRAPH_STYLE,
    LEFT_PANEL_CARD_STYLE,
    LEFT_PANEL_STYLE,
    MESH_CARD_STYLE,
    MESH_WRAPPER_STYLE,
    RIGHT_PANEL_STYLE,
    ROOT_STYLE,
    ROW_THREE_STYLE,
    ROW_TWO_STYLE,
    TAB_SELECTED_STYLE,
    TAB_STYLE,
)
from data.structures.three_d.mesh.mesh import Mesh
from data.viewer.utils.atomic_displays.image_display import create_image_display
from data.viewer.utils.atomic_displays.mesh_display import create_mesh_display

SCENE_RADIO_ID = "texture-benchmark-scene-radio"
RIGHT_PANEL_ID = "texture-benchmark-right-panel"


def build_root_layout(
    scene_names: List[str],
    default_scene_name: str,
    right_panel_children: List[Any],
) -> html.Div:
    """Build the root layout for the benchmark viewer.

    Args:
        scene_names: Available benchmark scene names.
        default_scene_name: Initial scene selection.
        right_panel_children: Initial right-panel component list.

    Returns:
        Root layout component.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(scene_names, list), (
            "Expected `scene_names` to be a list. " f"{type(scene_names)=}."
        )
        assert all(isinstance(scene_name, str) for scene_name in scene_names), (
            "Expected `scene_names` to contain only strings. " f"{scene_names=}"
        )
        assert isinstance(default_scene_name, str), (
            "Expected `default_scene_name` to be a string. "
            f"{type(default_scene_name)=}."
        )
        assert isinstance(right_panel_children, list), (
            "Expected `right_panel_children` to be a list. "
            f"{type(right_panel_children)=}."
        )

    _validate_inputs()

    scene_options = [
        {
            "label": scene_name,
            "value": scene_name,
        }
        for scene_name in scene_names
    ]
    return html.Div(
        style=ROOT_STYLE,
        children=[
            html.Div(
                style=LEFT_PANEL_STYLE,
                children=[
                    html.H1("Texture Extraction", style={"marginTop": "0"}),
                    html.P(
                        "Single-frame GSO-CLOD runtime and texture-quality benchmark."
                    ),
                    html.Div(
                        style=LEFT_PANEL_CARD_STYLE,
                        children=[
                            html.H3("Scene Selection", style={"marginTop": "0"}),
                            dcc.RadioItems(
                                id=SCENE_RADIO_ID,
                                options=scene_options,
                                value=default_scene_name,
                                labelStyle={
                                    "display": "block",
                                    "marginBottom": "10px",
                                    "fontSize": "14px",
                                },
                                inputStyle={"marginRight": "10px"},
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                id=RIGHT_PANEL_ID,
                style=RIGHT_PANEL_STYLE,
                children=right_panel_children,
            ),
        ],
    )


def build_scene_panel_children(
    scene_payload: Dict[str, Any],
) -> List[Any]:
    """Build the right-panel children for one cached scene payload.

    Args:
        scene_payload: Cached scene payload dictionary.

    Returns:
        Right-panel child component list.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(scene_payload, dict), (
            "Expected `scene_payload` to be a dict. " f"{type(scene_payload)=}."
        )
        assert "scene_name" in scene_payload, (
            "Expected `scene_payload` to contain `scene_name`. "
            f"{scene_payload.keys()=}"
        )

    _validate_inputs()

    return [
        html.Div(
            children=[
                html.H2(
                    scene_payload["scene_name"],
                    style={"marginBottom": "4px"},
                ),
                html.P(
                    "Three extraction methods, one first-frame source image, and per-scene timing breakdowns.",
                    style={"marginTop": "0"},
                ),
            ]
        ),
        dcc.Tabs(
            children=[
                dcc.Tab(
                    label="Visual Comparison",
                    style=TAB_STYLE,
                    selected_style=TAB_SELECTED_STYLE,
                    children=build_visual_tab(scene_payload=scene_payload),
                ),
                dcc.Tab(
                    label="Timing",
                    style=TAB_STYLE,
                    selected_style=TAB_SELECTED_STYLE,
                    children=build_timing_tab(scene_payload=scene_payload),
                ),
            ]
        ),
    ]


def build_visual_tab(
    scene_payload: Dict[str, Any],
) -> html.Div:
    """Build the visual-comparison tab for one scene.

    Args:
        scene_payload: Cached scene payload dictionary.

    Returns:
        Visual-tab component.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(scene_payload, dict), (
            "Expected `scene_payload` to be a dict. " f"{type(scene_payload)=}."
        )

    _validate_inputs()

    row_one = html.Div(
        style=ROW_TWO_STYLE,
        children=[
            _build_image_card(
                image_rgb=scene_payload["source_rgb"],
                title="Source Frame",
            ),
            _build_image_card(
                image_rgb=scene_payload["reference_texture_rgb"],
                title="Reference Texture",
            ),
        ],
    )
    row_two = html.Div(
        style=ROW_THREE_STYLE,
        children=[
            _build_method_texture_card(
                scene_payload=scene_payload,
                method_key="texel_visibility_v1",
            ),
            _build_method_texture_card(
                scene_payload=scene_payload,
                method_key="texel_visibility_v2",
            ),
            _build_method_texture_card(
                scene_payload=scene_payload,
                method_key="open3d_cpu",
            ),
        ],
    )
    row_three = html.Div(
        style=ROW_THREE_STYLE,
        children=[
            _build_method_mesh_card(
                scene_payload=scene_payload,
                method_key="texel_visibility_v1",
            ),
            _build_method_mesh_card(
                scene_payload=scene_payload,
                method_key="texel_visibility_v2",
            ),
            _build_method_mesh_card(
                scene_payload=scene_payload,
                method_key="open3d_cpu",
            ),
        ],
    )
    return html.Div(
        style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "16px",
            "padding": "12px 0",
        },
        children=[row_one, row_two, row_three],
    )


def build_timing_tab(
    scene_payload: Dict[str, Any],
) -> html.Div:
    """Build the timing tab for one scene.

    Args:
        scene_payload: Cached scene payload dictionary.

    Returns:
        Timing-tab component.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(scene_payload, dict), (
            "Expected `scene_payload` to be a dict. " f"{type(scene_payload)=}."
        )

    _validate_inputs()

    open3d_note = None
    if "open3d_gpu" not in scene_payload["methods"]:
        open3d_note = html.P(
            "Open3D GPU is omitted because the installed Open3D build does not support the required extraction operations on a non-CPU device here.",
            style={"margin": "0", "color": "#486581"},
        )
    timing_card_children: List[Any] = []
    if open3d_note is not None:
        timing_card_children.append(open3d_note)
    timing_card_children.append(
        dcc.Graph(
            figure=build_scene_timing_figure(scene_payload=scene_payload),
            style={"height": "540px"},
        )
    )
    return html.Div(
        style={"padding": "12px 0"},
        children=[html.Div(style=CARD_STYLE, children=timing_card_children)],
    )


def _build_image_card(
    image_rgb: Any,
    title: str,
) -> html.Div:
    """Build one image-display card.

    Args:
        image_rgb: RGB image array `[H, W, 3]`.
        title: Image-card title.

    Returns:
        Image-card component.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(title, str), (
            "Expected `title` to be a string. " f"{type(title)=}."
        )

    _validate_inputs()

    image_tensor = _image_rgb_to_chw_tensor(image_rgb=image_rgb)
    return html.Div(
        style=CARD_STYLE,
        children=[
            html.H3(title, style={"margin": "0"}),
            dcc.Graph(
                figure=create_image_display(image=image_tensor, title=title),
                style=GRAPH_STYLE,
            ),
        ],
    )


def _build_method_texture_card(
    scene_payload: Dict[str, Any],
    method_key: str,
) -> html.Div:
    """Build one extracted-texture card for one method.

    Args:
        scene_payload: Cached scene payload dictionary.
        method_key: Method identifier.

    Returns:
        Texture-card component.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(scene_payload, dict), (
            "Expected `scene_payload` to be a dict. " f"{type(scene_payload)=}."
        )
        assert isinstance(method_key, str), (
            "Expected `method_key` to be a string. " f"{type(method_key)=}."
        )
        assert method_key in scene_payload["methods"], (
            "Expected `method_key` to exist in `scene_payload['methods']`. "
            f"{method_key=} {scene_payload['methods'].keys()=}"
        )

    _validate_inputs()

    method_payload = scene_payload["methods"][method_key]
    image_tensor = _texture_map_to_chw_tensor(
        uv_texture_map=method_payload["uv_texture_map"]
    )
    return html.Div(
        style=CARD_STYLE,
        children=[
            html.H3(method_payload["display_label"], style={"margin": "0"}),
            dcc.Graph(
                figure=create_image_display(
                    image=image_tensor,
                    title=method_payload["display_label"],
                ),
                style=GRAPH_STYLE,
            ),
        ],
    )


def _build_method_mesh_card(
    scene_payload: Dict[str, Any],
    method_key: str,
) -> html.Div:
    """Build one textured-mesh card for one method.

    Args:
        scene_payload: Cached scene payload dictionary.
        method_key: Method identifier.

    Returns:
        Textured-mesh card component.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(scene_payload, dict), (
            "Expected `scene_payload` to be a dict. " f"{type(scene_payload)=}."
        )
        assert isinstance(method_key, str), (
            "Expected `method_key` to be a string. " f"{type(method_key)=}."
        )
        assert method_key in scene_payload["methods"], (
            "Expected `method_key` to exist in `scene_payload['methods']`. "
            f"{method_key=} {scene_payload['methods'].keys()=}"
        )

    _validate_inputs()

    method_payload = scene_payload["methods"][method_key]
    mesh = Mesh(
        vertices=scene_payload["mesh_vertices"],
        faces=scene_payload["mesh_faces"],
        uv_texture_map=method_payload["uv_texture_map"],
        vertex_uv=scene_payload["mesh_vertex_uv"],
        face_uvs=scene_payload["mesh_face_uvs"],
        convention="obj",
    )
    mesh_component = create_mesh_display(
        mesh=mesh,
        title=method_payload["display_label"],
        component_id=f"mesh-{method_key}",
        camera_sync_group="texture-benchmark-mesh-sync",
    )
    return html.Div(
        style=MESH_CARD_STYLE,
        children=[
            html.H3(method_payload["display_label"], style={"margin": "0"}),
            html.Div(style=MESH_WRAPPER_STYLE, children=[mesh_component]),
        ],
    )


def _image_rgb_to_chw_tensor(
    image_rgb: Any,
) -> torch.Tensor:
    """Convert one RGB image array into a CHW float32 tensor.

    Args:
        image_rgb: RGB image array `[H, W, 3]`.

    Returns:
        CHW float32 tensor `[3, H, W]` in `[0, 1]`.
    """

    image_tensor = torch.from_numpy(image_rgb).to(dtype=torch.float32).div(255.0)
    return image_tensor.permute(2, 0, 1).contiguous()


def _texture_map_to_chw_tensor(
    uv_texture_map: torch.Tensor,
) -> torch.Tensor:
    """Convert one HWC UV texture map into a CHW tensor for image display.

    Args:
        uv_texture_map: UV texture tensor `[H, W, 3]`.

    Returns:
        CHW float32 tensor `[3, H, W]`.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(uv_texture_map, torch.Tensor), (
            "Expected `uv_texture_map` to be a tensor. " f"{type(uv_texture_map)=}."
        )
        assert uv_texture_map.ndim == 3, (
            "Expected `uv_texture_map` to have shape `[H, W, 3]`. "
            f"{uv_texture_map.shape=}"
        )
        assert uv_texture_map.shape[2] == 3, (
            "Expected `uv_texture_map` to end in RGB channels. "
            f"{uv_texture_map.shape=}"
        )

    _validate_inputs()

    return uv_texture_map.permute(2, 0, 1).contiguous()
