"""Shared iframe-dispatch callback for reusable viewer apps.

Args:
    None.

Returns:
    None.
"""

from dash import ClientsideFunction, Dash, Input, Output


def register_iframe_dispatch_callback(
    app: Dash,
    input_store_id: str = "viewer-panels-store",
) -> None:
    """Register the shared clientside iframe-dispatch callback.

    Args:
        app: Dash application instance.
        input_store_id: Store id providing either the shared `viewer-panels-store`
            payload or a nested dispatch payload containing `viewer_panels`.

    Returns:
        None.
    """

    def _validate_register_iframe_dispatch_callback_inputs() -> None:
        assert isinstance(app, Dash), (
            "App must be a Dash instance. " f"app_type={type(app)}."
        )
        assert isinstance(input_store_id, str), (
            "Input store id must be a string. "
            f"input_store_id_type={type(input_store_id)}."
        )
        assert input_store_id != "", (
            "Input store id must be non-empty. " f"input_store_id={input_store_id!r}."
        )

    _validate_register_iframe_dispatch_callback_inputs()

    app.clientside_callback(
        ClientsideFunction(namespace="iframeDispatch", function_name="dispatch"),
        Output("iframe-dispatch-store", "data"),
        Input(input_store_id, "data"),
        prevent_initial_call=False,
    )
