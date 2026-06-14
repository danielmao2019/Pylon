"""Dash placeholder display API."""

from dash import html


def create_placeholder_display(message: str) -> html.Div:
    """Create a placeholder display.

    Args:
        message: Placeholder message.

    Returns:
        Dash placeholder display element.
    """
    assert isinstance(message, str), "Message must be a string. message=%r" % message
    return html.Div(message, className="placeholder-surface")
