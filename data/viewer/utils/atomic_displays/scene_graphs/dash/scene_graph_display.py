"""Dash scene-graph display API."""

from typing import Dict, List

from dash import html


def create_scene_graph_display(rows: List[Dict[str, str]]) -> html.Pre:
    """Create a Dash scene-graph display.

    Args:
        rows: Scene-graph preview rows.

    Returns:
        Dash scene-graph display element.
    """
    assert isinstance(rows, list), "Rows must be a list. rows=%r" % rows
    return html.Pre(str(rows), className="json-preview")
