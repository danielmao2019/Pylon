"""Dash text display API."""

from dash import html


def create_text_display(text: str) -> html.Pre:
    """Create a Dash text display.

    Args:
        text: Text content.

    Returns:
        Dash text display element.
    """
    assert isinstance(text, str), "Text must be a string. text=%r" % text
    return html.Pre(text, className="text-display")
