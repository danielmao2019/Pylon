from typing import Dict

DEFAULT_TITLE_MARGIN = 30
SMALL_FIGURE_TITLE_MARGIN = 25
MAIN_GRAPH_HEIGHT = '100%'
DEBUGGER_MAIN_GRAPH_HEIGHT = '60%'


def _figure_layout(title_margin_top: int) -> Dict[str, int | bool | Dict[str, int]]:
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
    return _figure_layout(margin_top)


def small_figure_layout() -> Dict[str, int | bool | Dict[str, int]]:
    return _figure_layout(SMALL_FIGURE_TITLE_MARGIN)


def coloraxis_no_scale() -> Dict[str, bool]:
    return {'showscale': False}


def main_graph_style(debugger_enabled: bool) -> Dict[str, str]:
    height = DEBUGGER_MAIN_GRAPH_HEIGHT if debugger_enabled else MAIN_GRAPH_HEIGHT
    return {'width': '100%', 'height': height}


def debugger_main_graph_style() -> Dict[str, str]:
    return main_graph_style(True)


def graph_style_full() -> Dict[str, str]:
    return main_graph_style(False)


def graph_style_full() -> Dict[str, str]:
    return {'width': '100%', 'height': '100%'}


def grid_graph_style() -> Dict[str, str]:
    return {'width': '100%', 'height': '100%', 'minHeight': '0'}


def toggle_button_style() -> Dict[str, str]:
    return {
        'padding': '6px 12px',
        'fontSize': '12px',
        'cursor': 'pointer',
        'border': '1px solid #6c757d',
        'backgroundColor': '#f8f9fa',
    }


def toggle_overlay_style() -> Dict[str, str]:
    return {
        'display': 'flex',
        'alignItems': 'center',
        'gap': '10px',
        'position': 'absolute',
        'top': '10px',
        'right': '10px',
        'padding': '8px 12px',
        'backgroundColor': 'rgba(255,255,255,0.9)',
        'borderRadius': '4px',
        'boxShadow': '0 1px 4px rgba(0,0,0,0.12)',
        'zIndex': 5,
    }


def status_text_style(debugger_enabled: bool) -> Dict[str, str]:
    return {
        'fontWeight': 'bold',
        'color': '#28a745' if debugger_enabled else '#6c757d',
    }


def container_style() -> Dict[str, str]:
    return {
        'width': '100%',
        'height': '100%',
        'display': 'flex',
        'flexDirection': 'column',
        'position': 'relative',
    }


def simple_body_style() -> Dict[str, str]:
    return {
        'flex': '1',
        'display': 'flex',
        'flexDirection': 'column',
    }


def half_panel_style() -> Dict[str, str]:
    return {'width': '50%', 'height': '100%'}


def info_panel_style() -> Dict[str, str]:
    return {
        'height': '40%',
        'padding': '10px',
        'backgroundColor': '#f8f9fa',
        'borderTop': '1px solid #dee2e6',
        'overflowY': 'auto',
    }


def anchor_summary_style() -> Dict[str, str]:
    return {
        'fontSize': '12px',
        'color': '#444',
        'marginBottom': '10px',
        'fontWeight': 'bold',
    }


def total_text_style() -> Dict[str, str]:
    return {
        'fontSize': '12px',
        'color': '#666',
        'marginBottom': '10px',
        'fontWeight': 'bold',
    }


def section_header_style() -> Dict[str, str]:
    return {'margin': '10px 0 5px 0', 'fontSize': '14px'}


def checklist_style() -> Dict[str, str]:
    return {'padding': '5px'}


def checklist_label_style() -> Dict[str, str]:
    return {'display': 'block', 'margin': '5px 0'}


def grid_style(grid_size: int) -> Dict[str, str]:
    return {
        'width': '50%',
        'height': '100%',
        'padding': '10px',
        'display': 'grid',
        'gridTemplateColumns': f'repeat({grid_size}, minmax(0, 1fr))',
        'gridAutoRows': '1fr',
        'gap': '10px',
        'boxSizing': 'border-box',
        'overflowY': 'auto',
        'alignItems': 'stretch',
    }


def row_style() -> Dict[str, str]:
    return {'width': '100%', 'height': '50%', 'display': 'flex'}


def debugger_layout_style() -> Dict[str, str]:
    return {
        'width': '100%',
        'height': '100%',
        'display': 'flex',
        'flexDirection': 'column',
    }
