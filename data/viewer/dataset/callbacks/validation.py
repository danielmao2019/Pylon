"""Trigger validation helper for Dash callbacks."""

import dash
from dash import exceptions


def validate_trigger(expected_id: str) -> None:
    # Input validations
    assert isinstance(
        expected_id, str
    ), f"expected_id must be str, got {type(expected_id)}"

    triggered = dash.callback_context.triggered
    if not triggered:
        raise exceptions.PreventUpdate
    triggered_id = triggered[0]["prop_id"].split(".")[0]
    if triggered_id != expected_id:
        raise exceptions.PreventUpdate
