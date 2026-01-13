from typing import Any, Callable, Dict, List, Optional, Union
import copy
import torch


def apply_tensor_op(
    func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    func_kwargs: Optional[Dict[str, Any]] = None,
    method: Optional[str] = None,
    method_kwargs: Optional[Dict[str, Any]] = None,
    inputs: Union[tuple, list, dict, torch.Tensor, float, Any] = None,
) -> Any:
    assert (func is not None) ^ (method is not None)
    if func_kwargs is not None:
        assert func is not None
    if method_kwargs is not None:
        assert method is not None
    if func is not None:
        kwargs = {} if func_kwargs is None else func_kwargs
        return _apply_tensor_func(func=func, func_kwargs=kwargs, inputs=inputs)
    if method is not None:
        kwargs = {} if method_kwargs is None else method_kwargs
        return _apply_tensor_method(method=method, method_kwargs=kwargs, inputs=inputs)
    assert 0


def _apply_tensor_func(
    func: Callable[[torch.Tensor], torch.Tensor],
    func_kwargs: Dict[str, Any],
    inputs: Union[tuple, list, dict, torch.Tensor, float, Any],
) -> Any:
    if isinstance(inputs, torch.Tensor):
        return func(inputs, **func_kwargs)
    if isinstance(inputs, tuple):
        return tuple(
            _apply_tensor_func(
                func=func, func_kwargs=func_kwargs, inputs=tuple_elem
            )
            for tuple_elem in inputs
        )
    if isinstance(inputs, list):
        return list(
            _apply_tensor_func(func=func, func_kwargs=func_kwargs, inputs=list_elem)
            for list_elem in inputs
        )
    if isinstance(inputs, dict):
        return {
            key: _apply_tensor_func(
                func=func, func_kwargs=func_kwargs, inputs=inputs[key]
            )
            for key in inputs.keys()
        }
    if hasattr(inputs, '__dict__'):
        cloned = copy.copy(inputs)
        for attr, value in vars(inputs).items():
            new_value = _apply_tensor_func(
                func=func, func_kwargs=func_kwargs, inputs=value
            )
            setattr(cloned, attr, new_value)
        return cloned
    return inputs


def _apply_tensor_method(
    method: str,
    method_kwargs: Dict[str, Any],
    inputs: Union[tuple, list, dict, torch.Tensor, float, Any],
) -> Any:
    if hasattr(inputs, method):
        method_ref = getattr(inputs, method)
        assert callable(method_ref)
        return method_ref(**method_kwargs)
    if isinstance(inputs, tuple):
        return tuple(
            _apply_tensor_method(
                method=method, method_kwargs=method_kwargs, inputs=tuple_elem
            )
            for tuple_elem in inputs
        )
    if isinstance(inputs, list):
        return list(
            _apply_tensor_method(
                method=method, method_kwargs=method_kwargs, inputs=list_elem
            )
            for list_elem in inputs
        )
    if isinstance(inputs, dict):
        return {
            key: _apply_tensor_method(
                method=method, method_kwargs=method_kwargs, inputs=inputs[key]
            )
            for key in inputs.keys()
        }
    if hasattr(inputs, '__dict__'):
        cloned = copy.copy(inputs)
        for attr, value in vars(inputs).items():
            new_value = _apply_tensor_method(
                method=method, method_kwargs=method_kwargs, inputs=value
            )
            setattr(cloned, attr, new_value)
        return cloned
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
