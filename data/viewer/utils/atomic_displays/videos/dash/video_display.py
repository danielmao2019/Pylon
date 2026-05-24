"""Dash video display API."""

from dash import html


def create_video_display(src: str | None, title: str) -> html.Div:
    """Create a Dash video display.

    Args:
        src: Optional video source URL.
        title: Video title.

    Returns:
        Dash video display element.
    """
    assert src is None or isinstance(src, str), (
        "Source must be None or a string. src=%r" % src
    )
    assert isinstance(title, str), "Title must be a string. title=%r" % title
    if src is None:
        return html.Div(
            "Placeholder for missing video.", className="placeholder-surface"
        )
    return html.Div(html.Video(src=src, controls=True, title=title))
