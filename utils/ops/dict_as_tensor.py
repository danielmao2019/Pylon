"""This module implements multiple utility functions to support the idea of treating a Python dictionary
as an any immutable object indexed tensor, of any data type. For now, we name dictionaries under this
view as 'buffer'. A buffer could be tuple, list, dict, numpy.ndarray, and torch.Tensor.
"""
from typing import Tuple, List, Sequence, Set, Dict, Any, Optional
import numpy
import torch


def buffer_equal(buffer: Any, other: Any) -> bool:
    result = True
    result = result and (type(buffer) == type(other) or set([type(buffer), type(other)]).issubset(set([int, float])))
    if type(buffer) in [tuple, list]:
        result = result and len(buffer) == len(other)
        result = result and all([buffer_equal(buffer[idx], other[idx]) for idx in range(len(buffer))])
    elif type(buffer) == dict:
        result = result and set(buffer.keys()) == set(other.keys())
        result = result and all([buffer_equal(buffer[key], other[key]) for key in buffer])
    elif type(buffer) == numpy.ndarray:
        result = result and numpy.array_equal(buffer, other)
    elif type(buffer) == torch.Tensor:
        result = result and torch.equal(buffer, other)
    else:
        result = result and buffer == other
    return result


def buffer_allclose(
    buffer, other,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    assert type(buffer) == type(other)
    if isinstance(buffer, (tuple, list)):
        assert len(buffer) == len(other), f"{len(buffer)=}, {len(other)=}"
        return all(
            buffer_allclose(buffer[idx], other[idx], rtol=rtol, atol=atol, equal_nan=equal_nan)
            for idx in range(len(buffer))
        )
    elif isinstance(buffer, dict):
        assert set(buffer.keys()) == set(other.keys()), f"{buffer.keys()=}, {other.keys()=}"
        return all(
            buffer_allclose(buffer[key], other[key], rtol=rtol, atol=atol, equal_nan=equal_nan)
            for key in buffer.keys()
        )
    elif isinstance(buffer, numpy.ndarray):
        assert buffer.shape == other.shape
        return numpy.allclose(buffer, other, rtol=rtol, atol=atol, equal_nan=equal_nan)
    elif isinstance(buffer, torch.Tensor):
        assert buffer.shape == other.shape, f"{buffer.shape=}, {other.shape=}"
        return torch.allclose(buffer, other, rtol=rtol, atol=atol, equal_nan=equal_nan)
    else:
        assert isinstance(buffer, (float, int)), f"{type(buffer)=}"
        return abs(buffer - other) <= atol + rtol * abs(other)


def _buffer_add(buffer: Any, other: Any) -> Any:
    assert type(buffer) == type(other) or set([type(buffer), type(other)]).issubset(set([int, float]))
    if type(buffer) in [tuple, list]:
        assert len(buffer) == len(other), f"{len(buffer)=}, {len(other)=}"
        return type(buffer)([_buffer_add(buffer[idx], other[idx]) for idx in range(len(buffer))])
    elif type(buffer) == dict:
        assert set(buffer.keys()) == set(other.keys()), f"{buffer.keys()=}, {other.keys()=}"
        return {key: _buffer_add(buffer[key], other[key]) for key in buffer}
    else:
        return buffer + other


def buffer_add(*buffers: Any) -> Any:
    result = buffers[0]
    for idx in range(1, len(buffers)):
        result = _buffer_add(result, buffers[idx])
    return result


def buffer_scalar_mul(buffer: Any, scalar: Any) -> Any:
    # input checks
    if type(scalar) == numpy.ndarray:
        assert scalar.size == 1, f"{scalar.shape=}"
        scalar = scalar.item()
    elif type(scalar) == torch.Tensor:
        assert scalar.numel == 1, f"{scalar.shape=}"
        scalar = scalar.item()
    else:
        assert type(scalar) in [int, float]
    # compute scalar multiplication
    if type(buffer) in [tuple, list]:
        return type(buffer)([buffer_scalar_mul(buffer[idx], scalar) for idx in range(len(buffer))])
    elif type(buffer) == dict:
        return {key: buffer_scalar_mul(buffer[key], scalar) for key in buffer}
    else:
        return buffer * scalar


def buffer_sub(buffer: Any, other: Any) -> Any:
    return buffer_add(buffer, buffer_scalar_mul(other, -1))


def buffer_mul(buffer: Any, other: Any) -> Any:
    assert type(buffer) == type(other) or set([type(buffer), type(other)]).issubset(set([int, float])), \
        f"{type(buffer)=}, {type(other)=}"
    if type(buffer) in [tuple, list]:
        assert len(buffer) == len(other), f"{len(buffer)=}, {len(other)=}"
        return type(buffer)([buffer_mul(buffer[idx], other[idx]) for idx in range(len(buffer))])
    elif type(buffer) == dict:
        assert set(buffer.keys()) == set(other.keys()), f"{buffer.keys()=}, {other.keys()=}"
        return {key: buffer_mul(buffer[key], other[key]) for key in buffer}
    else:
        return buffer * other


def buffer_rec(buffer: Any) -> Any:
    if type(buffer) in [tuple, list]:
        return type(buffer)([buffer_rec(buffer[idx]) for idx in range(len(buffer))])
    elif type(buffer) == dict:
        return {key: buffer_rec(buffer[key]) for key in buffer}
    else:
        return 1 / buffer


def buffer_div(buffer: Any, other: Any) -> Any:
    return buffer_mul(buffer, buffer_rec(other))


def buffer_mean(buffer: Sequence[Any]) -> Any:
    r"""Take the mean of the buffer along the first axis.
    """
    return buffer_scalar_mul(buffer_add(*list(buffer)), 1 / len(buffer))


def get_buffer_structure(buffer: Any) -> List[Tuple[Any, Set[Any]]]:
    """Get the structure of a buffer.
    """
    if isinstance(buffer, (list, tuple)):
        curr_structure = (type(buffer), set(range(len(buffer))))
        next_structures = [get_buffer_structure(elem) for elem in buffer]
    elif isinstance(buffer, dict):
        curr_structure = (type(buffer), set(buffer.keys()))
        next_structures = [get_buffer_structure(elem) for elem in buffer.values()]
    else:
        return []
    if len(next_structures) == 0:
        return [curr_structure]
    next_n_axes = max(list(map(len, next_structures)))
    next_structure = []
    for axis in range(next_n_axes):
        axis_types = set(
            elem_structures[axis][0] for elem_structures in next_structures
            if len(elem_structures) > axis
        )
        assert len(axis_types) == 1, f"{axis_types=}"
        axis_type = axis_types.pop()
        axis_indices = set.union(*[
            elem_structures[axis][1] for elem_structures in next_structures
            if len(elem_structures) > axis
        ])
        next_structure.append((axis_type, axis_indices))
    return [curr_structure] + next_structure


def buffer_select(buffer, axis: int, index: Any) -> Any:
    """Select a slice from a buffer at a specific axis and index.

    Args:
        buffer: A buffer (list, tuple, or dict) containing other buffers
        axis: The axis to slice at (0-based)
        index: The index to select at that axis

    Returns:
        The selected slice from the buffer
    """
    if axis == 0:
        return buffer[index]

    if isinstance(buffer, (list, tuple)):
        return type(buffer)(buffer_select(elem, axis - 1, index) for elem in buffer)
    elif isinstance(buffer, dict):
        return {k: buffer_select(v, axis - 1, index) for k, v in buffer.items()}
    else:
        raise NotImplementedError(f"Unsupported buffer type: {type(buffer)}")


def buffer_permute(
    buffer: Any,
    axes: Optional[Sequence[int]] = None,
    buffer_structure: Optional[List[Tuple[Any, Set[Any]]]] = None,
) -> Any:
    """Permute the axes of a buffer.

    Args:
        buffer: A buffer (list, tuple, or dict) containing other buffers
        axes: Optional sequence of ints specifying the permutation of axes.
              If None, uses the original order of axes.

    Returns:
        A new buffer with axes permuted according to the axes parameter.
    """

    if len(buffer) == 0:
        return buffer

    if axes is not None and axes == type(axes)(range(len(axes))):
        return buffer

    # Get the structure of the buffer
    structure = buffer_structure or get_buffer_structure(buffer)

    # Handle None case - use reverse order
    if axes is None:
        axes = list(range(len(structure)))[::-1]

    # Validate axes
    assert len(axes) == len(structure), f"{len(axes)=}, {len(structure)=}"
    assert min(axes) == 0, f"{min(axes)=}"
    assert max(axes) == len(axes) - 1, f"{max(axes)=}"
    assert set(axes) == set(range(len(axes))), f"{axes=}"

    # Get the target type and indices for the first axis after permutation
    target_type, target_indices = structure[axes[0]]

    # Create the result container
    if target_type in (list, tuple):
        # check if the indices are consecutive
        assert min(target_indices) == 0, f"{min(target_indices)=}"
        assert max(target_indices) == len(target_indices) - 1, f"{max(target_indices)=}"
        assert len(target_indices) == len(set(target_indices)), f"{target_indices=}"
        # For list/tuple, we'll collect values in a list first
        result = [None] * len(target_indices)
    else:  # dict
        result = {key: None for key in target_indices}

    # For each index in the target first axis
    for target_idx in target_indices:
        # Get the value by slicing at the correct axis
        value = buffer_select(buffer, axes[0], target_idx)

        # If this is the last axis, just copy the value
        if len(axes) == 1:
            result[target_idx] = value
        else:
            # Recursively permute the remaining axes
            # Create new axes list by removing the current axis and adjusting remaining indices
            remaining_axes = []
            for ax in axes[1:]:
                if ax > axes[0]:
                    remaining_axes.append(ax - 1)
                else:
                    remaining_axes.append(ax)
            # Create new structure by removing the current axis
            remaining_structure = [s for i, s in enumerate(structure) if i != axes[0]]
            result[target_idx] = buffer_permute(
                value,
                remaining_axes,
                buffer_structure=remaining_structure,
            )

    # Convert list to tuple if needed
    if target_type == tuple:
        result = tuple(result)

    return result


def transpose_buffer(buffer: List[Dict[Any, Any]]) -> Dict[Any, List[Any]]:
    """Legacy function that only handles List[Dict] -> Dict[List] transformation.
    Use buffer_permute for more general axis permutations.
    """
    structure = get_buffer_structure(buffer)
    assert len(structure) >= 2, f"Transpose is not supported for buffers with less than 2 axes."
    # For transpose, we swap the first two axes and keep the rest in order
    axes = (1, 0) + tuple(range(2, len(structure)))
    return buffer_permute(buffer, axes=axes, buffer_structure=structure)
