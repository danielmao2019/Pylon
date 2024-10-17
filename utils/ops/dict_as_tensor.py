"""This module implements multiple utility functions to support the idea of treating a Python dictionary
as an any immutable object indexed tensor, of any data type. For now, we name dictionaries under this
view as 'buffer'. A buffer could be tuple, list, dict, numpy.ndarray, and torch.Tensor.
"""
from typing import List, Dict, Union, Any
import numpy
import torch


def buffer_equal(buffer, other) -> bool:
    result = True
    result = result and type(buffer) == type(other)
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


def average_buffer(
    buffer: List[Dict[Any, Union[int, float, numpy.ndarray, torch.Tensor, list, dict]]],
) -> Dict[Any, Union[float, numpy.ndarray, torch.Tensor, list, dict]]:
    # input check
    assert type(buffer) == list, f"{type(buffer)=}"
    assert all(type(elem) == dict for elem in buffer)
    keys = buffer[0].keys()
    assert all(elem.keys() == keys for elem in buffer)
    dtypes = {key: type(buffer[0][key]) for key in keys}
    assert set(dtypes.values()).issubset(set([int, float, numpy.ndarray, torch.Tensor, list, dict]))
    assert all(all(type(elem[key]) == dtypes[key] for key in keys) for elem in buffer)
    # average buffer
    result: Dict[Any, Union[float, numpy.ndarray, torch.Tensor, list, dict]] = {}
    for key in keys:
        if dtypes[key] == numpy.ndarray:
            assert all(elem[key].shape == buffer[0][key].shape for elem in buffer)
            result[key] = numpy.mean(numpy.stack([elem[key] for elem in buffer], axis=0).astype(numpy.float32), axis=0)
        elif dtypes[key] == torch.Tensor:
            assert all(elem[key].shape == buffer[0][key].shape for elem in buffer)
            result[key] = torch.mean(torch.stack([elem[key] for elem in buffer], dim=0).type(torch.float32), dim=0)
        elif dtypes[key] == list:
            assert all(len(elem[key]) == len(buffer[0][key]) for elem in buffer)
            result[key] = numpy.mean(numpy.stack([numpy.array(elem[key]) for elem in buffer], axis=0).astype(numpy.float32), axis=0).tolist()
        elif dtypes[key] == dict:
            result[key] = average_buffer(buffer=[elem[key] for elem in buffer])
        else:
            result[key] = sum([elem[key] for elem in buffer]) / len(buffer)
    return result
