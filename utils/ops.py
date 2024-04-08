from typing import List, Union, Callable
import torch


def apply_tensor_op(
    func: Union[
        Callable[[torch.Tensor], torch.Tensor],
        Callable[[float], float],
    ],
    inputs: Union[tuple, list, dict, torch.Tensor, float],
) -> dict:
    if type(inputs) in [torch.Tensor, float]:
        return func(inputs)
    elif type(inputs) == tuple:
        return tuple(apply_tensor_op(func=func, inputs=x) for x in inputs)
    elif type(inputs) == list:
        return list(apply_tensor_op(func=func, inputs=x) for x in inputs)
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
