import math
from typing import Any, Dict, List, Optional

import torch
from dash import dcc, html

from data.viewer.utils.atomic_displays.image_display import create_image_display
from models.three_d.octree_gs import styles


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
    assert 'selected_levels_rgb' in render_outputs
    selected_levels_rgb = render_outputs['selected_levels_rgb']
    assert isinstance(selected_levels_rgb, list), f"{type(selected_levels_rgb)=}"

    assert 'selected_levels_density' in render_outputs
    selected_levels_density = render_outputs['selected_levels_density']
    assert isinstance(
        selected_levels_density, list
    ), f"{type(selected_levels_density)=}"

    assert 'total_levels' in render_outputs
    total_levels = int(render_outputs['total_levels'])
    assert total_levels > 0, f"{total_levels=}"

    assert 'gaussian_counts_per_level' in render_outputs
    gaussian_counts_per_level = render_outputs['gaussian_counts_per_level']
    assert isinstance(
        gaussian_counts_per_level, dict
    ), f"gaussian_counts_per_level should be a dict, got {type(gaussian_counts_per_level)=}"
    assert len(gaussian_counts_per_level) == total_levels, (
        f"gaussian_counts_per_level missing entries: "
        f"expected {total_levels} levels, got {len(gaussian_counts_per_level)}"
    )
    for level in range(total_levels):
        assert (
            level in gaussian_counts_per_level
        ), f"gaussian_counts_per_level missing level {level}"
        assert isinstance(
            gaussian_counts_per_level[level], int
        ), f"gaussian count for level {level} must be int, got {type(gaussian_counts_per_level[level])=}"

    assert 'total_gaussians' in render_outputs
    total_gaussians = render_outputs['total_gaussians']
    assert isinstance(total_gaussians, int), "total_gaussians required"

    assert 'anchor_mask_percentage' in render_outputs
    anchor_mask_percentage = render_outputs['anchor_mask_percentage']
    assert isinstance(anchor_mask_percentage, float), f"{type(anchor_mask_percentage)=}"

    assert 'anchor_mask_active' in render_outputs
    anchor_mask_active = render_outputs['anchor_mask_active']
    assert isinstance(anchor_mask_active, int), f"{type(anchor_mask_active)=}"

    assert 'anchor_mask_total' in render_outputs
    anchor_mask_total = render_outputs['anchor_mask_total']
    assert isinstance(anchor_mask_total, int), f"{type(anchor_mask_total)=}"

    assert 'density_image' in render_outputs
    density_image = render_outputs['density_image']
    assert isinstance(density_image, torch.Tensor), f"{type(density_image)=}"

    assert 'rgb_level_images' in render_outputs
    rgb_level_images = render_outputs['rgb_level_images']
    assert isinstance(rgb_level_images, list) and len(rgb_level_images) == total_levels

    assert 'density_level_images' in render_outputs
    density_level_images = render_outputs['density_level_images']
    assert (
        isinstance(density_level_images, list)
        and len(density_level_images) == total_levels
    )

    checkbox_options = _build_level_checkbox_options(
        gaussian_counts_per_level=gaussian_counts_per_level,
        total_levels=total_levels,
    )
    grid_size = math.ceil(math.sqrt(total_levels))
    anchor_summary = _build_anchor_mask_summary(
        anchor_mask_percentage=anchor_mask_percentage,
        anchor_mask_active=anchor_mask_active,
        anchor_mask_total=anchor_mask_total,
    )
    debugger_body = _build_debugger_layout(
        dataset_name=dataset_name,
        scene_name=scene_name,
        method_name=method_name,
        title=title,
        total_gaussians=total_gaussians,
        gaussian_counts_per_level=gaussian_counts_per_level,
        checkbox_options=checkbox_options,
        selected_levels_rgb=selected_levels_rgb,
        selected_levels_density=selected_levels_density,
        rgb_image=rgb_image,
        density_image=density_image,
        rgb_level_images=rgb_level_images,
        density_level_images=density_level_images,
        grid_size=grid_size,
        total_levels=total_levels,
        anchor_summary=anchor_summary,
    )
    return debugger_body


def _build_level_checkbox_options(
    gaussian_counts_per_level: Dict[int, int],
    total_levels: int,
) -> List[Dict[str, Any]]:
    return [
        {
            'label': f'Level {level} ({gaussian_counts_per_level[level]:,} gaussians)',
            'value': level,
        }
        for level in range(total_levels)
    ]


def _build_anchor_mask_summary(
    anchor_mask_percentage: Optional[float],
    anchor_mask_active: Optional[int],
    anchor_mask_total: Optional[int],
) -> Optional[html.Div]:
    if (
        anchor_mask_percentage is None
        or anchor_mask_active is None
        or anchor_mask_total is None
        or anchor_mask_total == 0
    ):
        return None
    formatted = (
        f"Anchor mask active: {anchor_mask_active:,}/{anchor_mask_total:,} "
        f"({anchor_mask_percentage:.2f}%)"
    )
    return html.Div(
        formatted,
        style=styles.anchor_summary_style(),
    )


def _build_debugger_layout(
    dataset_name: str,
    scene_name: str,
    method_name: str,
    title: str,
    total_gaussians: int,
    gaussian_counts_per_level: Dict[int, int],
    checkbox_options: List[Dict[str, Any]],
    selected_levels_rgb: List[int],
    selected_levels_density: List[int],
    rgb_image: torch.Tensor,
    density_image: torch.Tensor,
    rgb_level_images: List[torch.Tensor],
    density_level_images: List[torch.Tensor],
    grid_size: int,
    total_levels: int,
    anchor_summary: Optional[html.Div],
) -> html.Div:
    rgb_row = _build_modal_row(
        dataset_name=dataset_name,
        scene_name=scene_name,
        method_name=method_name,
        modality='rgb',
        title=title,
        main_image=rgb_image,
        level_images=rgb_level_images,
        gaussian_counts_per_level=gaussian_counts_per_level,
        checkbox_id={
            'type': 'octree-levels-checklist-rgb',
            'dataset': dataset_name,
            'scene': scene_name,
            'method': method_name,
        },
        checkbox_options=checkbox_options,
        selected_levels=selected_levels_rgb,
        total_gaussians=total_gaussians,
        grid_size=grid_size,
        title_prefix="RGB",
        total_levels=total_levels,
        anchor_summary=anchor_summary,
    )
    density_row = _build_modal_row(
        dataset_name=dataset_name,
        scene_name=scene_name,
        method_name=method_name,
        modality='density',
        title=title,
        main_image=density_image,
        level_images=density_level_images,
        gaussian_counts_per_level=gaussian_counts_per_level,
        checkbox_id={
            'type': 'octree-levels-checklist-density',
            'dataset': dataset_name,
            'scene': scene_name,
            'method': method_name,
        },
        checkbox_options=checkbox_options,
        selected_levels=selected_levels_density,
        total_gaussians=total_gaussians,
        grid_size=grid_size,
        title_prefix="Density",
        total_levels=total_levels,
        anchor_summary=anchor_summary,
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
    level_images: List[torch.Tensor],
    gaussian_counts_per_level: Dict[int, int],
    checkbox_id: Dict[str, Any],
    checkbox_options: List[Dict[str, Any]],
    selected_levels: List[int],
    total_gaussians: int,
    grid_size: int,
    title_prefix: str,
    total_levels: int,
    anchor_summary: Optional[html.Div],
) -> html.Div:
    left_section = _build_modal_section(
        dataset_name=dataset_name,
        scene_name=scene_name,
        method_name=method_name,
        modality=modality,
        image=main_image,
        title=title,
        header_text=f"{title_prefix} - Octree Levels:",
        total_gaussians=total_gaussians,
        checkbox_id=checkbox_id,
        checkbox_options=checkbox_options,
        selected_levels=selected_levels,
        anchor_summary=anchor_summary,
    )
    right_section = _build_level_grid(
        level_images=level_images,
        gaussian_counts_per_level=gaussian_counts_per_level,
        grid_size=grid_size,
        title_prefix=title_prefix,
        total_levels=total_levels,
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
    selected_levels: List[int],
    anchor_summary: Optional[html.Div],
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
        selected_levels=selected_levels,
        anchor_summary=anchor_summary,
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
    selected_levels: List[int],
    anchor_summary: Optional[html.Div],
) -> html.Div:
    info_children: List[Any] = [
        html.H6(
            header_text,
            style=styles.section_header_style(),
        ),
        html.Div(
            f"Total: {total_gaussians:,} gaussians",
            style=styles.total_text_style(),
        ),
    ]
    if anchor_summary is not None:
        info_children.append(anchor_summary)
    info_children.append(
        dcc.Checklist(
            id=checkbox_id,
            options=checkbox_options,
            value=selected_levels,
            style=styles.checklist_style(),
            labelStyle=styles.checklist_label_style(),
        )
    )
    return html.Div(
        info_children,
        style=styles.info_panel_style(),
    )


def _build_level_grid(
    level_images: List[torch.Tensor],
    gaussian_counts_per_level: Dict[int, int],
    grid_size: int,
    title_prefix: str,
    total_levels: int,
) -> html.Div:
    grid_elements: List[dcc.Graph] = []
    for level in range(total_levels):
        level_figure = _build_level_figure(
            image=level_images[level],
            level=level,
            gaussian_count=gaussian_counts_per_level[level],
            title_prefix=title_prefix,
        )
        grid_elements.append(
            dcc.Graph(
                figure=level_figure,
                style=styles.grid_graph_style(),
                config={'responsive': True, 'displayModeBar': False},
            )
        )
    return html.Div(
        grid_elements,
        style=styles.grid_style(grid_size),
    )


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
            'type': 'octree-main-image',
            'modality': modality,
            'dataset': dataset_name,
            'scene': scene_name,
            'method': method_name,
        },
        figure=figure,
        style=style,
        config={'responsive': True, 'displayModeBar': False},
    )


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


def _build_level_figure(
    image: torch.Tensor,
    level: int,
    gaussian_count: int,
    title_prefix: str,
) -> Any:
    figure = create_image_display(
        image=image,
        title=f"{title_prefix} L{level} ({gaussian_count:,})",
        colorscale="Viridis",
    )
    figure.update_layout(**styles.small_figure_layout())
    figure.update_coloraxes(**styles.coloraxis_no_scale())
    return figure
