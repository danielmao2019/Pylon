from typing import List, Dict, Union, Callable, Any
import numpy
import torch


def apply_tensor_op(
    func: Callable[[torch.Tensor], torch.Tensor],
    inputs: Union[tuple, list, dict, torch.Tensor, float],
) -> dict:
    if type(inputs) == torch.Tensor:
        return func(inputs)
    elif type(inputs) == tuple:
        return tuple(apply_tensor_op(func=func, inputs=tuple_elem) for tuple_elem in inputs)
    elif type(inputs) == list:
        return list(apply_tensor_op(func=func, inputs=list_elem) for list_elem in inputs)
    elif type(inputs) == dict:
        return {key: apply_tensor_op(func=func, inputs=inputs[key]) for key in inputs.keys()}
    else:
        return inputs


def apply_pairwise(
    lot: List[torch.Tensor],
    func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    symmetric: bool = False,
) -> torch.Tensor:
    # input checks
    assert type(lot) == list, f"{type(lot)=}"
    # compute result
    dim = len(lot)
    result = torch.zeros(size=(dim, dim), dtype=torch.float32, device=torch.device('cuda'))
    for i in range(dim):
        loop = range(i, dim) if symmetric else range(dim)
        for j in loop:
            val = func(lot[i], lot[j])
            assert val.numel() == 1, f"{val.numel()}"
            result[i, j] = val
            if symmetric:
                result[j, i] = val
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
