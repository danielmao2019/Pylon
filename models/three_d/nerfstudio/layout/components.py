from typing import Any, Dict

import torch
from dash import dcc, html

from data.viewer.utils.atomic_displays.image_display import create_image_display
from models.three_d.nerfstudio import styles


def build_display(render_outputs: Dict[str, Any]) -> html.Div:
    assert 'image' in render_outputs, "render_outputs must include 'image'"
    image = render_outputs['image']
    assert isinstance(image, torch.Tensor), f"{type(image)=}"

    assert 'title' in render_outputs, "render_outputs must include 'title'"
    title = render_outputs['title']
    assert isinstance(title, str), f"{type(title)=}"

    return _build_display_main(image=image, title=title)


def _build_display_main(image: torch.Tensor, title: str) -> html.Div:
    return html.Div(
        [_build_main_graph(image=image, title=title)],
        style=styles.container_style(),
    )


def _build_main_graph(image: torch.Tensor, title: str) -> dcc.Graph:
    figure = _build_main_figure(image=image, title=title)
    return dcc.Graph(
        figure=figure,
        style=styles.graph_style(),
        config={'responsive': True, 'displayModeBar': False},
    )


def _build_main_figure(image: torch.Tensor, title: str) -> Any:
    figure = create_image_display(
        image=image,
        title=title,
        colorscale="Viridis",
    )
    figure.update_layout(**styles.figure_layout_with_title(title))
    figure.update_coloraxes(**styles.coloraxis_no_scale())
    return figure
