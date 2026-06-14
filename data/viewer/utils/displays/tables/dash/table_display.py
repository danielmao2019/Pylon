"""Dash table display API."""

from typing import Dict, List

from dash import dash_table


def create_table_display(rows: List[Dict[str, str]]) -> dash_table.DataTable:
    """Create a Dash table display.

    Args:
        rows: Table rows.

    Returns:
        Dash table display element.
    """
    assert isinstance(rows, list), "Rows must be a list. rows=%r" % rows
    columns = sorted({key for row in rows for key in row})
    return dash_table.DataTable(
        data=rows,
        columns=[{"name": column, "id": column} for column in columns],
    )
