from typing import Dict

DEFAULT_TITLE_MARGIN = 40


def figure_layout(title_margin_top: int) -> Dict[str, int | bool | Dict[str, int]]:
    return {
        'autosize': False,
        'margin': {'l': 0, 'r': 0, 't': title_margin_top, 'b': 0},
        'coloraxis_showscale': False,
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
    }


def figure_layout_with_title(title: str) -> Dict[str, int | bool | Dict[str, int]]:
    assert isinstance(title, str), f"{type(title)=}"
    margin_top = DEFAULT_TITLE_MARGIN if title else 0
    return figure_layout(margin_top)


def graph_style() -> Dict[str, str]:
    return {'width': '100%', 'height': '100%'}


def container_style() -> Dict[str, str]:
    return {'width': '100%', 'height': '100%'}


def coloraxis_no_scale() -> Dict[str, bool]:
    return {'showscale': False}
