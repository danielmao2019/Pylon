from typing import List, Union, Callable, Optional, Any
import torch


def apply_tensor_op(
    func: Callable[[torch.Tensor], torch.Tensor],
    inputs: Union[tuple, list, dict, torch.Tensor, float],
) -> Any:
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


def apply_op(
    func: Callable[[Any], Any],
    inputs: Any,
) -> Any:
    """Apply a function recursively to nested data structures.
    
    Recursively applies func to all non-container elements in nested
    tuples, lists, and dictionaries. If the input is not a container
    (tuple, list, dict), applies func directly to it.
    
    Args:
        func: Function to apply to non-container elements
        inputs: Input data structure (can be nested)
        
    Returns:
        Data structure with same nesting as inputs, but with func applied
        to all non-container elements
    """
    if type(inputs) == tuple:
        return tuple(apply_op(func=func, inputs=tuple_elem) for tuple_elem in inputs)
    elif type(inputs) == list:
        return list(apply_op(func=func, inputs=list_elem) for list_elem in inputs)
    elif type(inputs) == dict:
        return {key: apply_op(func=func, inputs=inputs[key]) for key in inputs.keys()}
    else:
        return func(inputs)


def apply_pairwise(
    func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    inputs: List[torch.Tensor],
    symmetric: Optional[bool] = False,
) -> torch.Tensor:
    # input checks
    assert type(inputs) == list, f"{type(inputs)=}"
    assert all([type(elem) == torch.Tensor for elem in inputs])
    # initialization
    device = inputs[0].device
    dim = len(inputs)
    # compute result
    result = torch.zeros(size=(dim, dim), dtype=torch.float32, device=device)
    for i in range(dim):
        loop = range(i, dim) if symmetric else range(dim)
        for j in loop:
            val = func(inputs[i], inputs[j])
            assert val.numel() == 1, f"{val.numel()}"
            result[i, j] = val
            if symmetric:
                result[j, i] = val
    return result
