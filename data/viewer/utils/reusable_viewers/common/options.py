"""Dash option helpers for shared viewer selectors."""

from typing import Dict, List, Optional


def make_options(values: List[str]) -> List[Dict[str, str]]:
    """Build Dash dropdown options from raw values.

    Args:
        values: Selector values in desired display order.

    Returns:
        Dash option dictionaries using the same label and value.
    """
    assert isinstance(values, list), "Values must be a list. values=%r" % (values,)
    options = []
    for value in values:
        assert isinstance(value, str), "Option value must be a string. value=%r" % (
            value,
        )
        options.append({"label": value, "value": value})
    return options


def first_option_value(options: List[Dict[str, str]]) -> str:
    """Return the first value from a non-empty option list.

    Args:
        options: Dash option dictionaries.

    Returns:
        The first option value.
    """
    assert isinstance(options, list), "Options must be a list. options=%r" % (options,)
    assert options, "Options must be non-empty. options=%r" % options
    first_option = options[0]
    assert "value" in first_option, "Option is missing value. option=%r" % (
        first_option,
    )
    return first_option["value"]


def choose_available_value(
    requested_value: Optional[str],
    available_values: List[str],
    fallback_value: Optional[str] = None,
) -> Optional[str]:
    """Choose a valid value from the available set.

    Args:
        requested_value: Requested selector value.
        available_values: Ordered available values.
        fallback_value: Optional persisted fallback value.

    Returns:
        Resolved value, or `None` when nothing is available.
    """
    assert requested_value is None or isinstance(requested_value, str), (
        "Requested value must be None or a string. requested_value=%r"
        % (requested_value,)
    )
    assert isinstance(available_values, list), (
        "Available values must be a list. available_values=%r" % (available_values,)
    )
    assert fallback_value is None or isinstance(fallback_value, str), (
        "Fallback value must be None or a string. fallback_value=%r"
        % (fallback_value,)
    )

    if requested_value in available_values:
        return requested_value
    if fallback_value in available_values:
        return fallback_value
    if len(available_values) == 0:
        return None
    return available_values[0]
