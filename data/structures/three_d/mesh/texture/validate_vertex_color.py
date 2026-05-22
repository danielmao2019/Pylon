from typing import Any

import torch


def validate_vertex_color(obj: Any) -> None:
    assert isinstance(obj, torch.Tensor), (
        "Expected `vertex_color` to be a `torch.Tensor`. " f"{type(obj)=}"
    )
    assert obj.ndim in (2, 3), (
        "Expected `vertex_color` to be rank 2 or 3. " f"{obj.shape=}"
    )
    if obj.ndim == 3:
        assert obj.shape[0] == 1, (
            "Expected rank-3 `vertex_color` to have batch size 1. " f"{obj.shape=}"
        )
        obj = obj[0]
    assert obj.shape[1] == 3, (
        "Expected `vertex_color` to have RGB values in the last dimension. "
        f"{obj.shape=}"
    )
    assert obj.shape[0] > 0, (
        "Expected `vertex_color` to contain at least one RGB row. " f"{obj.shape=}"
    )

    if obj.dtype == torch.uint8:
        _validate_vertex_color_uint8(obj=obj)
        return
    if obj.dtype == torch.float32:
        _validate_vertex_color_float32(obj=obj)
        return
    raise AssertionError(
        "Expected `vertex_color` to be either uint8 `[0, 255]` or "
        "float32 `[0, 1]`. "
        f"{obj.dtype=}"
    )


def _validate_vertex_color_uint8(obj: Any) -> None:
    assert obj.dtype == torch.uint8, (
        "Expected `vertex_color` uint8 validation to receive uint8 values. "
        f"{obj.dtype=}"
    )


def _validate_vertex_color_float32(obj: Any) -> None:
    assert obj.dtype == torch.float32, (
        "Expected `vertex_color` float32 validation to receive float32 "
        "values. "
        f"{obj.dtype=}"
    )
    assert torch.isfinite(obj).all(), (
        "Expected float32 `vertex_color` to contain only finite RGB values. "
        f"{obj.shape=} {obj.dtype=}"
    )
    min_value = float(obj.min().item())
    max_value = float(obj.max().item())
    assert min_value >= 0.0, (
        "Expected float32 `vertex_color` RGB values to be at least 0. " f"{min_value=}"
    )
    assert max_value <= 1.0, (
        "Expected float32 `vertex_color` RGB values to be at most 1. " f"{max_value=}"
    )
