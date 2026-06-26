import importlib

import pytest
import torch

# ====================================================================================================
from utils.ops.chunked_matmul import chunked_matmul

# Import the module object directly: the package __init__ rebinds the name
# `chunked_matmul` to the function, shadowing the submodule on attribute access,
# so `importlib.import_module` is used to reach the true module for monkeypatching.
chunked_matmul_module = importlib.import_module("utils.ops.chunked_matmul")

# ====================================================================================================


@pytest.mark.parametrize(
    "N, K, M, num_divide",
    [
        (1, 4, 3, 0),
        (10, 8, 5, 1),
        (10, 8, 5, 2),
        (37, 16, 7, 3),
        (100, 32, 9, 4),
    ],
)
def test_matches_plain_matmul(N: int, K: int, M: int, num_divide: int) -> None:
    """chunked_matmul equals large @ small across several N and num_divide splits."""
    large = torch.randn(N, K, dtype=torch.float64)
    small = torch.randn(K, M, dtype=torch.float64)
    expected = large @ small
    result = chunked_matmul(large=large, small=small, num_divide=num_divide)
    assert result.shape == (N, M), f"unexpected shape {result.shape=} vs {(N, M)=}"
    assert torch.allclose(
        result, expected
    ), f"chunked product differs from plain matmul, {num_divide=}"


@pytest.mark.parametrize("num_divide", [0, 2])
def test_supports_autograd(num_divide: int) -> None:
    """chunked_matmul backpropagates and its grads match a plain large @ small across num_divide splits."""
    large = torch.randn(10, 8, dtype=torch.float64, requires_grad=True)
    small = torch.randn(8, 5, dtype=torch.float64, requires_grad=True)

    ref_large = large.detach().clone().requires_grad_(True)
    ref_small = small.detach().clone().requires_grad_(True)

    result = chunked_matmul(large=large, small=small, num_divide=num_divide)
    expected = ref_large @ ref_small
    assert torch.allclose(
        result, expected
    ), f"chunked forward differs from plain matmul, {num_divide=}"

    result.sum().backward()
    expected.sum().backward()

    assert large.grad is not None, f"large.grad should not be None, {num_divide=}"
    assert small.grad is not None, f"small.grad should not be None, {num_divide=}"
    assert torch.allclose(
        large.grad, ref_large.grad
    ), f"large.grad differs from plain-matmul grad, {num_divide=}"
    assert torch.allclose(
        small.grad, ref_small.grad
    ), f"small.grad differs from plain-matmul grad, {num_divide=}"


def test_halves_batch_on_cuda_oom(monkeypatch: pytest.MonkeyPatch) -> None:
    """a first-batch CUDA OOM triggers a halving retry that then completes correctly."""
    large = torch.randn(20, 6, dtype=torch.float64)
    small = torch.randn(6, 4, dtype=torch.float64)
    expected = large @ small

    real_batched = chunked_matmul_module._chunked_matmul_batched
    state = {"calls": 0}

    def fake_batched(
        large: torch.Tensor, small: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        state["calls"] += 1
        if state["calls"] == 1:
            assert (
                batch_size == large.shape[0]
            ), f"first attempt should use full batch, got {batch_size=}"
            raise torch.cuda.OutOfMemoryError("simulated OOM on first batch")
        return real_batched(large=large, small=small, batch_size=batch_size)

    monkeypatch.setattr(chunked_matmul_module, "_chunked_matmul_batched", fake_batched)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    result = chunked_matmul(large=large, small=small, max_divide=2)
    assert state["calls"] == 2, f"expected exactly one retry, got {state['calls']=}"
    assert torch.allclose(
        result, expected
    ), "result after halving retry differs from plain matmul"


def test_raises_after_max_divide_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    """OOM persisting past max_divide raises torch.cuda.OutOfMemoryError."""
    large = torch.randn(20, 6, dtype=torch.float64)
    small = torch.randn(6, 4, dtype=torch.float64)

    def always_oom(
        large: torch.Tensor, small: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        raise torch.cuda.OutOfMemoryError("simulated persistent OOM")

    monkeypatch.setattr(chunked_matmul_module, "_chunked_matmul_batched", always_oom)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    with pytest.raises(torch.cuda.OutOfMemoryError):
        chunked_matmul(large=large, small=small, max_divide=2)


@pytest.mark.parametrize(
    "large, small",
    [
        (torch.randn(4), torch.randn(4, 3)),
        (torch.randn(2, 4, 3), torch.randn(3, 5)),
        (torch.randn(4, 3), torch.randn(3)),
        (torch.randn(4, 3), torch.randn(2, 3, 5)),
    ],
)
def test_rejects_non_2d_operands(large: torch.Tensor, small: torch.Tensor) -> None:
    """a vector, batched, or N-D large or small raises an assertion (both operands must be 2D)."""
    with pytest.raises(AssertionError):
        chunked_matmul(large=large, small=small)
