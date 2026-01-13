import math
from typing import Any, Dict, List

import torch
from dash import dcc, html

from data.viewer.utils.atomic_displays.image_display import create_image_display
from models.three_d.lapis_gs import styles


# Tree (DFS-ordered)
def build_display(render_outputs: Dict[str, Any]) -> html.Div:
    assert 'dataset_name' in render_outputs, "render_outputs missing dataset_name"
    dataset_name = render_outputs['dataset_name']
    assert isinstance(dataset_name, str), f"{type(dataset_name)=}"

    assert 'scene_name' in render_outputs, "render_outputs missing scene_name"
    scene_name = render_outputs['scene_name']
    assert isinstance(scene_name, str), f"{type(scene_name)=}"

    assert 'method_name' in render_outputs, "render_outputs missing method_name"
    method_name = render_outputs['method_name']
    assert isinstance(method_name, str), f"{type(method_name)=}"

    assert 'debugger_enabled' in render_outputs, "render_outputs missing debugger flag"
    debugger_enabled = bool(render_outputs['debugger_enabled'])

    assert 'rgb_image' in render_outputs, "render_outputs missing rgb_image"
    rgb_image = render_outputs['rgb_image']
    assert isinstance(rgb_image, torch.Tensor), f"{type(rgb_image)=}"

    assert 'title' in render_outputs, "render_outputs missing title"
    title = render_outputs['title']
    assert isinstance(title, str), f"{type(title)=}"

    if not debugger_enabled:
        return _build_display_main(
            dataset_name=dataset_name,
            scene_name=scene_name,
            method_name=method_name,
            image=rgb_image,
            title=title,
        )

    return _build_display_debugger(
        render_outputs=render_outputs,
        dataset_name=dataset_name,
        scene_name=scene_name,
        method_name=method_name,
        rgb_image=rgb_image,
        title=title,
    )


def _build_display_main(
    dataset_name: str,
    scene_name: str,
    method_name: str,
    image: torch.Tensor,
    title: str,
) -> html.Div:
    rgb_graph = _build_main_graph(
        dataset_name=dataset_name,
        scene_name=scene_name,
        method_name=method_name,
        image=image,
        title=title,
        modality='rgb',
        debugger_enabled=False,
    )
    return html.Div(
        [rgb_graph],
        style=styles.simple_body_style(),
    )


def _build_display_debugger(
    render_outputs: Dict[str, Any],
    dataset_name: str,
    scene_name: str,
    method_name: str,
    rgb_image: torch.Tensor,
    title: str,
) -> html.Div:
    assert 'selected_layers_rgb' in render_outputs
    selected_layers_rgb = render_outputs['selected_layers_rgb']
    assert isinstance(selected_layers_rgb, list), f"{type(selected_layers_rgb)=}"

    assert 'selected_layers_density' in render_outputs
    selected_layers_density = render_outputs['selected_layers_density']
    assert isinstance(
        selected_layers_density, list
    ), f"{type(selected_layers_density)=}"

    assert 'num_layers' in render_outputs
    num_layers = int(render_outputs['num_layers'])
    assert num_layers > 0, f"{num_layers=}"

    assert 'layer_names' in render_outputs
    layer_names = render_outputs['layer_names']
    assert isinstance(layer_names, list) and len(layer_names) == num_layers

    assert 'gaussian_counts_per_layer' in render_outputs
    gaussian_counts_per_layer = render_outputs['gaussian_counts_per_layer']
    assert isinstance(
        gaussian_counts_per_layer, list
    ), "gaussian_counts_per_layer required"

    assert 'total_gaussians' in render_outputs
    total_gaussians = render_outputs['total_gaussians']
    assert isinstance(total_gaussians, int), "total_gaussians required"

    assert 'density_image' in render_outputs
    density_image = render_outputs['density_image']
    assert isinstance(density_image, torch.Tensor), f"{type(density_image)=}"

    assert 'rgb_layer_images' in render_outputs
    rgb_layer_images = render_outputs['rgb_layer_images']
    assert isinstance(rgb_layer_images, list) and len(rgb_layer_images) == num_layers

    assert 'density_layer_images' in render_outputs
    density_layer_images = render_outputs['density_layer_images']
    assert (
        isinstance(density_layer_images, list)
        and len(density_layer_images) == num_layers
    )

    checkbox_options = _build_layer_checkbox_options(
        layer_names=layer_names,
        gaussian_counts_per_layer=gaussian_counts_per_layer,
        num_layers=num_layers,
    )
    grid_size = math.ceil(math.sqrt(num_layers))
    debugger_body = _build_debugger_layout(
        dataset_name=dataset_name,
        scene_name=scene_name,
        method_name=method_name,
        title=title,
        total_gaussians=total_gaussians,
        layer_names=layer_names,
        gaussian_counts_per_layer=gaussian_counts_per_layer,
        checkbox_options=checkbox_options,
        selected_layers_rgb=selected_layers_rgb,
        selected_layers_density=selected_layers_density,
        rgb_image=rgb_image,
        density_image=density_image,
        rgb_layer_images=rgb_layer_images,
        density_layer_images=density_layer_images,
        grid_size=grid_size,
        num_layers=num_layers,
    )
    return debugger_body


def _build_layer_checkbox_options(
    layer_names: List[str],
    gaussian_counts_per_layer: List[int],
    num_layers: int,
) -> List[Dict[str, Any]]:
    return [
        {
            'label': f'{layer_names[layer]} ({gaussian_counts_per_layer[layer]:,} gaussians)',
            'value': layer,
        }
        for layer in range(num_layers)
    ]


def _build_debugger_layout(
    dataset_name: str,
    scene_name: str,
    method_name: str,
    title: str,
    total_gaussians: int,
    layer_names: List[str],
    gaussian_counts_per_layer: List[int],
    checkbox_options: List[Dict[str, Any]],
    selected_layers_rgb: List[int],
    selected_layers_density: List[int],
    rgb_image: torch.Tensor,
    density_image: torch.Tensor,
    rgb_layer_images: List[torch.Tensor],
    density_layer_images: List[torch.Tensor],
    grid_size: int,
    num_layers: int,
) -> html.Div:
    rgb_row = _build_modal_row(
        dataset_name=dataset_name,
        scene_name=scene_name,
        method_name=method_name,
        modality='rgb',
        title=title,
        main_image=rgb_image,
        layer_images=rgb_layer_images,
        layer_names=layer_names,
        gaussian_counts_per_layer=gaussian_counts_per_layer,
        checkbox_id={
            'type': 'lapis-layers-checklist-rgb',
            'dataset': dataset_name,
            'scene': scene_name,
            'method': method_name,
        },
        checkbox_options=checkbox_options,
        selected_layers=selected_layers_rgb,
        total_gaussians=total_gaussians,
        grid_size=grid_size,
        title_prefix="RGB",
        num_layers=num_layers,
    )
    density_row = _build_modal_row(
        dataset_name=dataset_name,
        scene_name=scene_name,
        method_name=method_name,
        modality='density',
        title=title,
        main_image=density_image,
        layer_images=density_layer_images,
        layer_names=layer_names,
        gaussian_counts_per_layer=gaussian_counts_per_layer,
        checkbox_id={
            'type': 'lapis-layers-checklist-density',
            'dataset': dataset_name,
            'scene': scene_name,
            'method': method_name,
        },
        checkbox_options=checkbox_options,
        selected_layers=selected_layers_density,
        total_gaussians=total_gaussians,
        grid_size=grid_size,
        title_prefix="Density",
        num_layers=num_layers,
    )
    return html.Div(
        [rgb_row, density_row],
        style=styles.debugger_layout_style(),
    )


def _build_modal_row(
    dataset_name: str,
    scene_name: str,
    method_name: str,
    modality: str,
    title: str,
    main_image: torch.Tensor,
    layer_images: List[torch.Tensor],
    layer_names: List[str],
    gaussian_counts_per_layer: List[int],
    checkbox_id: Dict[str, Any],
    checkbox_options: List[Dict[str, Any]],
    selected_layers: List[int],
    total_gaussians: int,
    grid_size: int,
    title_prefix: str,
    num_layers: int,
) -> html.Div:
    left_section = _build_modal_section(
        dataset_name=dataset_name,
        scene_name=scene_name,
        method_name=method_name,
        modality=modality,
        image=main_image,
        title=title,
        header_text=f"{title_prefix} - LapisGS Layers:",
        total_gaussians=total_gaussians,
        checkbox_id=checkbox_id,
        checkbox_options=checkbox_options,
        selected_layers=selected_layers,
    )
    right_section = _build_layer_grid(
        layer_images=layer_images,
        layer_names=layer_names,
        gaussian_counts_per_layer=gaussian_counts_per_layer,
        grid_size=grid_size,
        title_prefix=title_prefix,
        num_layers=num_layers,
    )
    return html.Div(
        [left_section, right_section],
        style=styles.row_style(),
    )


def _build_modal_section(
    dataset_name: str,
    scene_name: str,
    method_name: str,
    modality: str,
    image: torch.Tensor,
    title: str,
    header_text: str,
    total_gaussians: int,
    checkbox_id: Dict[str, Any],
    checkbox_options: List[Dict[str, Any]],
    selected_layers: List[int],
) -> html.Div:
    main_graph = _build_main_graph(
        dataset_name=dataset_name,
        scene_name=scene_name,
        method_name=method_name,
        image=image,
        title=title,
        modality=modality,
        debugger_enabled=True,
    )
    info_panel = _build_info_panel(
        header_text=header_text,
        total_gaussians=total_gaussians,
        checkbox_id=checkbox_id,
        checkbox_options=checkbox_options,
        selected_layers=selected_layers,
    )
    return html.Div(
        [main_graph, info_panel],
        style=styles.half_panel_style(),
    )


def _build_info_panel(
    header_text: str,
    total_gaussians: int,
    checkbox_id: Dict[str, Any],
    checkbox_options: List[Dict[str, Any]],
    selected_layers: List[int],
) -> html.Div:
    return html.Div(
        [
            html.H6(
                header_text,
                style=styles.section_header_style(),
            ),
            html.Div(
                f"Total: {total_gaussians:,} gaussians",
                style=styles.total_text_style(),
            ),
            dcc.Checklist(
                id=checkbox_id,
                options=checkbox_options,
                value=selected_layers,
                style=styles.checklist_style(),
                labelStyle=styles.checklist_label_style(),
            ),
        ],
        style=styles.info_panel_style(),
    )


def _build_layer_grid(
    layer_images: List[torch.Tensor],
    layer_names: List[str],
    gaussian_counts_per_layer: List[int],
    grid_size: int,
    title_prefix: str,
    num_layers: int,
) -> html.Div:
    grid_elements: List[dcc.Graph] = []
    for layer in range(num_layers):
        layer_figure = _build_layer_figure(
            image=layer_images[layer],
            title=f"{title_prefix} {layer_names[layer]} ({gaussian_counts_per_layer[layer]:,})",
        )
        grid_elements.append(
            dcc.Graph(
                figure=layer_figure,
                style=styles.grid_graph_style(),
                config={'responsive': True, 'displayModeBar': False},
            )
        )
    return html.Div(
        grid_elements,
        style=styles.grid_style(grid_size),
    )


def _build_layer_figure(
    image: torch.Tensor,
    title: str,
) -> Any:
    figure = create_image_display(
        image=image,
        title=title,
        colorscale="Viridis",
    )
    figure.update_layout(**styles.small_figure_layout())
    figure.update_coloraxes(**styles.coloraxis_no_scale())
    return figure


def _build_main_figure(
    image: torch.Tensor,
    title: str,
    modality: str,
) -> Any:
    modality_title = modality.upper()
    figure_title = f"{title} - {modality_title}" if title else modality_title
    figure = create_image_display(
        image=image,
        title=figure_title,
        colorscale="Viridis",
    )
    figure.update_layout(**styles.figure_layout_with_title(title))
    figure.update_coloraxes(**styles.coloraxis_no_scale())
    return figure


# Shared helpers (used by multiple parents)
def _build_main_graph(
    dataset_name: str,
    scene_name: str,
    method_name: str,
    image: torch.Tensor,
    title: str,
    modality: str,
    debugger_enabled: bool,
) -> dcc.Graph:
    figure = _build_main_figure(image=image, title=title, modality=modality)
    style = (
        styles.debugger_main_graph_style()
        if debugger_enabled
        else styles.graph_style_full()
    )
    return dcc.Graph(
        id={
            'type': 'lapis-main-image',
            'modality': modality,
            'dataset': dataset_name,
            'scene': scene_name,
            'method': method_name,
        },
        figure=figure,
        style=style,
        config={'responsive': True, 'displayModeBar': False},
    )
