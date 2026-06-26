import math
from typing import Optional

import torch


def _validate_inputs(
    large: torch.Tensor, small: torch.Tensor, max_divide: int, num_divide: Optional[int]
) -> None:
    """Validate the operands and division controls for chunked matmul.

    Args:
        large: Left operand. Must be a 2D torch.Tensor of shape [N, K], any floating dtype.
        small: Right operand. Must be a 2D torch.Tensor of shape [K, M], any floating dtype.
        max_divide: Maximum number of halving retries on CUDA OOM. Must be an int >= 0.
        num_divide: Optional fixed number of halvings. If not None, must be an int >= 0.

    Returns:
        None.
    """
    assert isinstance(
        large, torch.Tensor
    ), f"large must be a torch.Tensor, got {type(large)=}"
    assert isinstance(
        small, torch.Tensor
    ), f"small must be a torch.Tensor, got {type(small)=}"
    assert (
        large.ndim == 2
    ), f"large must be a 2D tensor, got {large.ndim=} with {large.shape=}"
    assert (
        small.ndim == 2
    ), f"small must be a 2D tensor, got {small.ndim=} with {small.shape=}"
    assert (
        large.shape[1] == small.shape[0]
    ), f"inner dimensions must match for matmul, got {large.shape=} and {small.shape=}"
    assert (
        large.device == small.device
    ), f"operands must be on the same device, got {large.device=} and {small.device=}"
    assert (
        large.dtype == small.dtype
    ), f"operands must share the same dtype, got {large.dtype=} and {small.dtype=}"
    assert isinstance(
        max_divide, int
    ), f"max_divide must be an int, got {type(max_divide)=}"
    assert max_divide >= 0, f"max_divide must be non-negative, got {max_divide=}"
    assert num_divide is None or isinstance(
        num_divide, int
    ), f"num_divide must be None or an int, got {type(num_divide)=}"
    assert (
        num_divide is None or num_divide >= 0
    ), f"num_divide must be non-negative when set, got {num_divide=}"


def _normalize_inputs(large: torch.Tensor, small: torch.Tensor) -> torch.Tensor:
    """Make the right operand contiguous so chunked matmul writes are well-defined.

    Args:
        large: Left operand of shape [N, K] (returned unchanged; only validated upstream).
        small: Right operand of shape [K, M], any floating dtype.

    Returns:
        small: The right operand made contiguous, shape [K, M], same dtype/device as input.
    """
    return small.contiguous()


def _chunked_matmul_batched(
    large: torch.Tensor, small: torch.Tensor, batch_size: int
) -> torch.Tensor:
    """Multiply `large` by `small` in row-chunks of `batch_size`.

    Args:
        large: Left operand of shape [N, K], any floating dtype, on the device used for the product.
        small: Right operand of shape [K, M], same dtype/device as `large`.
        batch_size: Positive int number of rows of `large` processed per chunk.

    Returns:
        result: The [N, M] product, same dtype/device as `large`.
    """
    N = large.shape[0]
    M = small.shape[1]
    result = torch.empty((N, M), dtype=large.dtype, device=large.device)
    for i in range(0, N, batch_size):
        result[i : i + batch_size] = large[i : i + batch_size] @ small
    return result


def chunked_matmul(
    large: torch.Tensor,
    small: torch.Tensor,
    max_divide: int = 0,
    num_divide: Optional[int] = None,
) -> torch.Tensor:
    """Multiply a large 2D tensor by a small 2D tensor on its right, chunking the large's first dim.

    When `num_divide` is set, the large's first dim is split into a fixed number of halvings and
    multiplied in one batched pass. Otherwise the batch is progressively halved on CUDA OOM, up to
    `max_divide` halvings, releasing cached CUDA memory between attempts.

    Args:
        large: Left operand of shape [N, K], any floating dtype.
        small: Right operand of shape [K, M], same dtype/device as `large`.
        max_divide: Maximum number of halving retries on CUDA OOM. int >= 0.
        num_divide: Optional fixed number of halvings. int >= 0 when set, else None.

    Returns:
        The [N, M] product, same dtype/device as `large`.
    """
    _validate_inputs(
        large=large, small=small, max_divide=max_divide, num_divide=num_divide
    )
    small = _normalize_inputs(large=large, small=small)

    N = large.shape[0]
    if num_divide is not None:
        bs = max(1, math.ceil(N / 2**num_divide))
        return _chunked_matmul_batched(large=large, small=small, batch_size=bs)

    n = 0
    while n <= max_divide:
        bs = max(1, math.ceil(N / 2**n))
        try:
            return _chunked_matmul_batched(large=large, small=small, batch_size=bs)
        except torch.cuda.OutOfMemoryError:
            n += 1
            torch.cuda.empty_cache()
            continue
        except Exception:
            raise

    raise torch.cuda.OutOfMemoryError(
        f"CUDA OOM after {max_divide} halvings in chunked_matmul, with {large.shape=} and {small.shape=}"
    )
