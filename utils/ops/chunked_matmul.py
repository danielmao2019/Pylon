import math
from typing import Optional

import torch


def _validate_inputs(
    large: torch.Tensor,
    small: torch.Tensor,
    inplace: bool,
    max_divide: int,
    num_divide: Optional[int],
) -> None:
    """Validate the operands and controls for chunked matmul.

    Args:
        large: Left operand. Must be a 2D torch.Tensor of shape [N, K], any floating dtype.
        small: Right operand. Must be a 2D square torch.Tensor of shape [K, K], same dtype and device as large.
        inplace: Whether the product overwrites large. Must be a bool; when True, neither operand may require grad (an in-place overwrite of large is illegal under autograd).
        max_divide: Maximum number of chunk halvings on CUDA OOM. Must be an int >= 0.
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
    assert small.shape[0] == small.shape[1], f"small must be square, got {small.shape=}"
    assert (
        large.shape[1] == small.shape[0]
    ), f"inner dimensions must match for matmul, got {large.shape=} and {small.shape=}"
    assert (
        large.device == small.device
    ), f"operands must be on the same device, got {large.device=} and {small.device=}"
    assert (
        large.dtype == small.dtype
    ), f"operands must share the same dtype, got {large.dtype=} and {small.dtype=}"
    assert isinstance(inplace, bool), f"inplace must be a bool, got {type(inplace)=}"
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
    if inplace:
        assert (
            not large.requires_grad and not small.requires_grad
        ), f"inplace=True overwrites large and is illegal under autograd, got {large.requires_grad=} and {small.requires_grad=}"


def _matmul_chunk(
    large: torch.Tensor, small: torch.Tensor, out: torch.Tensor, direct: bool
) -> None:
    """Write the product large @ small into out for one row-chunk.

    Args:
        large: Left operand chunk of shape [b, K], any floating dtype.
        small: Right square operand of shape [K, K], same dtype and device as large.
        out: Destination chunk of shape [b, K], same dtype and device as large; may alias large's rows only when direct is False.
        direct: When True the GEMM writes straight into out with no intermediate (out must be a distinct, non-grad buffer); when False a temp-copy assignment is used (autograd-safe, and the only correct form when out aliases large, since a GEMM whose out aliases an operand is undefined behavior).

    Returns:
        None.
    """
    if direct:
        torch.matmul(large, small, out=out)
    else:
        out[:] = large @ small


def chunked_matmul(
    large: torch.Tensor,
    small: torch.Tensor,
    inplace: bool = False,
    max_divide: int = 0,
    num_divide: Optional[int] = None,
) -> torch.Tensor:
    """Multiply a large 2D tensor by a small square 2D tensor on its right, chunking the large's first dim.

    The chunk size is ceil(N / 2 ** num_divide) when num_divide is set, otherwise it starts at N and is halved on each CUDA OOM up to max_divide times, releasing cached CUDA memory between attempts. The loop is resume-safe: a halving continues from the first not-yet-written chunk and never recomputes a completed one, so the in-place path can never double-transform an already-written row. Peak memory follows three paths: inplace overwrites large with no output allocation (only a per-chunk intermediate); the not-inplace no-grad path writes each chunk straight into the output (output only, no intermediate); the not-inplace grad path index-assigns each chunk (output plus a per-chunk intermediate, the autograd-safe minimum).

    Args:
        large: Left operand of shape [N, K], any floating dtype.
        small: Right operand of shape [K, K] (square), same dtype and device as large.
        inplace: When True the product overwrites large and large is returned; requires that neither operand requires grad.
        max_divide: Maximum number of chunk halvings on CUDA OOM. int >= 0.
        num_divide: Optional fixed number of halvings, with no OOM retry. int >= 0 when set, else None.

    Returns:
        The [N, K] product, same dtype and device as large; the large object itself when inplace.
    """
    _validate_inputs(
        large=large,
        small=small,
        inplace=inplace,
        max_divide=max_divide,
        num_divide=num_divide,
    )
    small = small.contiguous()

    N = large.shape[0]
    M = small.shape[1]
    out = (
        large
        if inplace
        else torch.empty((N, M), dtype=large.dtype, device=large.device)
    )
    direct = not inplace and not large.requires_grad and not small.requires_grad

    bs = max(1, math.ceil(N / 2**num_divide)) if num_divide is not None else N
    i = 0
    divides = 0
    while i < N:
        j = min(N, i + bs)
        try:
            _matmul_chunk(large=large[i:j], small=small, out=out[i:j], direct=direct)
        except torch.cuda.OutOfMemoryError:
            if num_divide is not None or divides >= max_divide or bs <= 1:
                raise
            divides += 1
            bs = max(1, bs // 2)
            torch.cuda.empty_cache()
            continue
        i = j
    return out
