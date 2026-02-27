from typing import Callable, Dict

import torch


def materialize_tensor(
    tensor: torch.Tensor,
    method: str = "cpu_numpy",
) -> torch.Tensor:
    """Materialize a tensor using a selected backend method."""
    assert isinstance(
        tensor, torch.Tensor
    ), f"Input must be torch.Tensor, got {type(tensor)}"
    assert isinstance(method, str), f"{type(method)=}"

    method_to_fn: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "clone_detach": _materialize_tensor_clone_detach,
        "cpu_numpy": _materialize_tensor_cpu_numpy,
    }
    assert method in method_to_fn, (
        f"Unsupported materialize method: {method}. "
        f"Supported methods: {list(method_to_fn.keys())}"
    )

    materialized_tensor = method_to_fn[method](tensor=tensor)
    assert isinstance(
        materialized_tensor, torch.Tensor
    ), f"{type(materialized_tensor)=}"
    assert (
        materialized_tensor.shape == tensor.shape
    ), f"{materialized_tensor.shape=}, {tensor.shape=}"
    assert (
        materialized_tensor.dtype == tensor.dtype
    ), f"{materialized_tensor.dtype=}, {tensor.dtype=}"
    assert (
        materialized_tensor.device == tensor.device
    ), f"{materialized_tensor.device=}, {tensor.device=}"
    return materialized_tensor


def _materialize_tensor_clone_detach(tensor: torch.Tensor) -> torch.Tensor:
    # Input validations
    assert isinstance(
        tensor, torch.Tensor
    ), f"Input must be torch.Tensor, got {type(tensor)}"
    return tensor.clone().detach()


def _materialize_tensor_cpu_numpy(tensor: torch.Tensor) -> torch.Tensor:
    # Input validations
    assert isinstance(
        tensor, torch.Tensor
    ), f"Input must be torch.Tensor, got {type(tensor)}"

    detached_tensor = tensor.detach()
    tensor_numpy = detached_tensor.cpu().numpy()
    return torch.tensor(
        tensor_numpy,
        dtype=tensor.dtype,
        device=tensor.device,
    )
