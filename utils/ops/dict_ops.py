"""Generic dictionary operations."""

from copy import deepcopy
from typing import Any, Dict


def merge_missing_dict_values(
    input_dict: Dict[str, Any],
    default_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Fill missing nested dictionary values without overwriting provided values.

    Args:
        input_dict: User-provided dictionary whose values take precedence.
        default_dict: Default-value dictionary with the same nested structure.

    Returns:
        New dictionary containing `input_dict` values plus any missing defaults.
    """

    def _validate_inputs() -> None:
        """Validate merge inputs.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(input_dict, dict), (
            "Expected `input_dict` to be a `dict`. " f"{type(input_dict)=}"
        )
        assert isinstance(default_dict, dict), (
            "Expected `default_dict` to be a `dict`. " f"{type(default_dict)=}"
        )

    _validate_inputs()

    merged_dict: Dict[str, Any] = {}
    for key, default_value in default_dict.items():
        if key in input_dict:
            input_value = input_dict[key]
            if isinstance(default_value, dict):
                merged_dict[key] = merge_missing_dict_values(
                    input_dict=input_value,
                    default_dict=default_value,
                )
                continue
            merged_dict[key] = deepcopy(input_value)
            continue
        merged_dict[key] = deepcopy(default_value)

    for key, input_value in input_dict.items():
        if key not in merged_dict:
            merged_dict[key] = deepcopy(input_value)

    return merged_dict
