"""Layout components for the Agents viewer Dash app."""

from typing import Any, Dict, List

import dash
from dash import dash_table, dcc, html

from agents.viewer.layout.styles import (
    LAST_UPDATE_STYLE,
    PROGRESS_STYLE,
    TABLE_CELL_STYLE,
    TABLE_DATA_STYLE,
    TABLE_HEADER_STYLE,
)


def build_layout(
    app: dash.Dash,
    last_update_text: str,
    progress_text: str,
    table_data: List[Dict[str, Any]],
    table_style: List[Dict[str, Any]],
) -> None:
    # Input validations
    assert isinstance(app, dash.Dash), f"app must be Dash, got {type(app)}"
    assert isinstance(
        last_update_text, str
    ), f"last_update_text must be str, got {type(last_update_text)}"
    assert isinstance(
        progress_text, str
    ), f"progress_text must be str, got {type(progress_text)}"
    assert isinstance(table_data, list), f"table_data must be list, got {type(table_data)}"
    assert isinstance(
        table_style, list
    ), f"table_style must be list, got {type(table_style)}"

    layout = html.Div(
        [
            html.H1("Server System Status Dashboard"),
            html.Div(
                id="last-update",
                children=last_update_text,
                style=LAST_UPDATE_STYLE,
            ),
            html.Div(
                id="progress",
                children=progress_text,
                style=PROGRESS_STYLE,
            ),
            dcc.Interval(id="interval-component", interval=2 * 1000, n_intervals=0),
            dash_table.DataTable(
                id="status-table",
                columns=[
                    {"name": "Server", "id": "Server"},
                    {"name": "Resource", "id": "Resource"},
                    {"name": "Utilization", "id": "Utilization"},
                    {"name": "Free Memory", "id": "Free Memory"},
                    {"name": "User", "id": "User"},
                    {"name": "PID", "id": "PID"},
                    {"name": "Start", "id": "Start"},
                    {"name": "CMD", "id": "CMD"},
                ],
                data=table_data,
                merge_duplicate_headers=True,
                style_cell=TABLE_CELL_STYLE,
                style_header=TABLE_HEADER_STYLE,
                style_data=TABLE_DATA_STYLE,
                style_data_conditional=table_style,
            ),
        ]
    )
    app.layout = layout
