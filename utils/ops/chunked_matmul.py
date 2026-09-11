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
        small.ndim >= 2
    ), f"small must be at least 2D, got {small.ndim=} with {small.shape=}"
    assert (
        small.shape[-2] == small.shape[-1]
    ), f"small must be square in its trailing two axes, got {small.shape=}"
    assert (
        large.shape[1] == small.shape[-2]
    ), f"inner dimensions must match for matmul, got {large.shape=} and {small.shape=}"
    assert (
        large.device == small.device
    ), f"operands must be on the same device, got {large.device=} and {small.device=}"
    assert (
        large.dtype == small.dtype
    ), f"operands must share the same dtype, got {large.dtype=} and {small.dtype=}"
    assert isinstance(inplace, bool), f"inplace must be a bool, got {type(inplace)=}"
    assert (
        isinstance(max_divide, int) and max_divide >= 0
    ), f"max_divide must be a non-negative int, got {type(max_divide)=} {max_divide=}"
    assert num_divide is None or (
        isinstance(num_divide, int) and num_divide >= 0
    ), f"num_divide must be None or a non-negative int, got {type(num_divide)=} {num_divide=}"
    if inplace:
        assert (
            small.ndim == 2
        ), f"inplace=True requires a 2D small: leading axes make the product wider than large, leaving nothing to overwrite in place, got {small.shape=}"
        assert (
            not large.requires_grad and not small.requires_grad
        ), f"inplace=True overwrites large and is illegal under autograd, got {large.requires_grad=} and {small.requires_grad=}"


def _matmul_chunk(
    large: torch.Tensor, small: torch.Tensor, out: torch.Tensor, direct: bool
) -> None:
    """Write the product large @ small into out for one row-chunk, as one plain 2-D product per [K, K] entry of small's leading axes.

    Args:
        large: Left operand chunk of shape [b, K], any floating dtype.
        small: Contiguous right square operand of shape [..., K, K], same dtype and device as large; each [K, K] entry of its leading axes multiplies large on its own, and an unbatched [K, K] small is its single entry.
        out: Destination chunk of shape [..., b, M], same dtype and device as large, whose leading axes match small's; each [b, M] entry receives the product with the matching small entry; may alias large's rows only when direct is False.
        direct: When True the GEMM writes straight into out with no intermediate (out must be a distinct, non-grad buffer); when False a temp-copy assignment is used (autograd-safe, and the only correct form when out aliases large, since a GEMM whose out aliases an operand is undefined behavior).

    Returns:
        None.
    """
    # One plain product per entry: CUDA's batched product rounds unlike the unbatched one at some row counts, and each entry must match what it gives multiplied alone. view addresses the entries without a copy, and each entry is indexed on its own so the autograd path may write into it in place.
    small_entries = small.view(-1, *small.shape[-2:])
    out_entries = out.view(-1, *out.shape[-2:])
    for index in range(small_entries.shape[0]):
        if direct:
            torch.matmul(large, small_entries[index], out=out_entries[index])
        else:
            out_entries[index][:] = large @ small_entries[index]


def chunked_matmul(
    large: torch.Tensor,
    small: torch.Tensor,
    inplace: bool = False,
    max_divide: int = 0,
    num_divide: Optional[int] = None,
) -> torch.Tensor:
    """Multiply a large 2D tensor by a small [..., K, K] tensor on its right, chunking the large's first dim.

    The chunk size is ceil(N / 2 ** num_divide) when num_divide is set, otherwise it starts at N and is halved on each CUDA OOM up to max_divide times, releasing cached CUDA memory between attempts. The loop is resume-safe: a halving continues from the first not-yet-written chunk and never recomputes a completed one, so the in-place path can never double-transform an already-written row. Peak memory follows three paths: inplace overwrites large with no output allocation (only a per-chunk intermediate); the not-inplace no-grad path writes each chunk straight into the output (output only, no intermediate); the not-inplace grad path index-assigns each chunk (output plus a per-chunk intermediate, the autograd-safe minimum).

    Args:
        large: Left operand of shape [N, K], any floating dtype.
        small: Right operand of shape [..., K, K] (square in its trailing two axes), same dtype and device as large. A batched small broadcasts large over its leading axes, so every matrix it carries multiplies the same [N, K] large.
        inplace: When True the product overwrites large and large is returned; requires a 2D small (leading axes make the product wider than large) and that neither operand requires grad. A CUDA-OOM that exhausts max_divide mid-pass leaves large partially transformed (already-written chunks are not rolled back, since that would need the inverse of small); use inplace=False for all-or-nothing semantics.
        max_divide: Maximum number of chunk halvings on CUDA OOM. int >= 0.
        num_divide: Optional fixed number of halvings, with no OOM retry. int >= 0 when set, else None.

    Returns:
        The [..., N, M] product carrying small's leading axes, same dtype and device as large; the large object itself when inplace, which a batched small therefore cannot produce.
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
    M = small.shape[-1]
    out = (
        large
        if inplace
        else torch.empty(
            tuple(small.shape[:-2]) + (N, M), dtype=large.dtype, device=large.device
        )
    )
    direct = not inplace and not large.requires_grad and not small.requires_grad

    bs = max(1, math.ceil(N / 2**num_divide)) if num_divide is not None else N
    i = 0
    divides = 0
    while i < N:
        j = min(N, i + bs)
        try:
            _matmul_chunk(
                large=large[i:j], small=small, out=out[..., i:j, :], direct=direct
            )
        except torch.cuda.OutOfMemoryError:
            if num_divide is not None or divides >= max_divide or bs <= 1:
                raise
            divides += 1
            bs = max(1, bs // 2)
            torch.cuda.empty_cache()
            continue
        i = j
    return out
