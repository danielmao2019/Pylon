"""This module implements multiple utility functions to support the idea of treating a Python dictionary
as an any immutable object indexed tensor, of any data type. For now, we name dictionaries under this
view as 'buffer'. A buffer could be tuple, list, dict, numpy.ndarray, and torch.Tensor.
"""
from typing import List, Dict, Any
import numpy
import torch


def buffer_equal(buffer, other) -> bool:
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


def _buffer_add(buffer, other):
    assert type(buffer) == type(other) or set([type(buffer), type(other)]).issubset(set([int, float]))
    if type(buffer) in [tuple, list]:
        assert len(buffer) == len(other), f"{len(buffer)=}, {len(other)=}"
        return type(buffer)([_buffer_add(buffer[idx], other[idx]) for idx in range(len(buffer))])
    elif type(buffer) == dict:
        assert set(buffer.keys()) == set(other.keys()), f"{buffer.keys()=}, {other.keys()=}"
        return {key: _buffer_add(buffer[key], other[key]) for key in buffer}
    else:
        return buffer + other


def buffer_add(*buffers):
    result = buffers[0]
    for idx in range(1, len(buffers)):
        result = _buffer_add(result, buffers[idx])
    return result


def buffer_scalar_mul(buffer, scalar):
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


def buffer_sub(buffer, other):
    return buffer_add(buffer, buffer_scalar_mul(other, -1))


def buffer_mul(buffer, other):
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


def buffer_rec(buffer):
    if type(buffer) in [tuple, list]:
        return type(buffer)([buffer_rec(buffer[idx]) for idx in range(len(buffer))])
    elif type(buffer) == dict:
        return {key: buffer_rec(buffer[key]) for key in buffer}
    else:
        return 1 / buffer


def buffer_div(buffer, other):
    return buffer_mul(buffer, buffer_rec(other))


def buffer_mean(buffer):
    r"""Take the mean of the buffer along the first axis.
    """
    return buffer_scalar_mul(buffer_add(*list(buffer)), 1 / len(buffer))


def transpose_buffer(buffer: List[Dict[Any, Any]]) -> Dict[Any, List[Any]]:
    # input check
    assert type(buffer) == list, f"{type(buffer)=}"
    assert all(type(elem) == dict for elem in buffer)
    keys = buffer[0].keys()
    assert all(elem.keys() == keys for elem in buffer)
    # transpose buffer
    result: Dict[Any, List[Any]] = {
        key: [elem[key] for elem in buffer]
        for key in keys
    }
    return result
