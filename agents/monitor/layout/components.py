"""Layout components for the Agents monitor Dash app."""

from typing import Any, Dict, List

from dash import Dash, dash_table, dcc, html

from agents.monitor.dashboard import build_columns
from agents.monitor.layout.styles import (
    CONTAINER_STYLE,
    HEADER_STYLE,
    LAST_UPDATE_STYLE,
    META_STYLE,
    TABLE_DATA_STYLE,
    TABLE_HEADER_STYLE,
    TABLE_STYLE,
)


def build_layout(
    app: Dash,
    meta_text: str,
    initial_rows: List[Dict[str, Any]],
    interval_ms: int,
) -> None:
    # Input validations
    assert isinstance(app, Dash), f"app must be Dash, got {type(app)}"
    assert isinstance(meta_text, str), f"meta_text must be str, got {type(meta_text)}"
    assert isinstance(
        initial_rows, list
    ), f"initial_rows must be list, got {type(initial_rows)}"
    assert isinstance(
        interval_ms, int
    ), f"interval_ms must be int, got {type(interval_ms)}"

    columns = build_columns()

    layout = html.Div(
        [
            html.H2("Agents Monitor Dashboard", style=HEADER_STYLE),
            html.Div(meta_text, id="monitor-meta", style=META_STYLE),
            html.Div(id="last-update", style=LAST_UPDATE_STYLE),
            dash_table.DataTable(
                id="resource-table",
                columns=columns,
                data=initial_rows,
                style_cell=TABLE_STYLE,
                style_header=TABLE_HEADER_STYLE,
                style_data=TABLE_DATA_STYLE,
            ),
            dcc.Interval(
                id="refresh-interval", interval=interval_ms, n_intervals=0
            ),
        ],
        style=CONTAINER_STYLE,
    )
    app.layout = layout
