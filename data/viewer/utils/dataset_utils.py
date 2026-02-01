"""Utility functions for the dataset viewer."""

import json
from typing import Any

import torch

from data.structures.three_d.camera.camera import Camera


def _normalize_for_display(value: Any) -> Any:
    if isinstance(value, Camera):
        return value.to_serialized()
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={list(value.shape)}, dtype={value.dtype})"
    if isinstance(value, dict):
        return {key: _normalize_for_display(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_for_display(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_for_display(item) for item in value]
    if isinstance(value, set):
        return [_normalize_for_display(item) for item in value]
    return value


def format_value(value: Any) -> str:
    """Format a value for display in error messages."""
    normalized = _normalize_for_display(value)
    if isinstance(normalized, (dict, list)):
        return json.dumps(normalized, indent=2, default=str)
    return str(normalized)
