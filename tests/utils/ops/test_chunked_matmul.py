import importlib
from typing import Optional

import pytest
import torch

from utils.ops.chunked_matmul import chunked_matmul

# Import the module object directly to monkeypatch its module-level `_matmul_chunk`:
# the package __init__ rebinds the name `chunked_matmul` to the function, shadowing
# the submodule on attribute access, so import_module reaches the true module.
chunked_matmul_module = importlib.import_module("utils.ops.chunked_matmul")


@pytest.mark.parametrize(
    "N, K, num_divide",
    [
        (1, 3, None),
        (10, 5, 0),
        (10, 5, 1),
        (10, 5, 2),
        (37, 7, 3),
        (100, 9, 4),
    ],
)
def test_matches_plain_matmul(N: int, K: int, num_divide: Optional[int]) -> None:
    """not-inplace no-grad path (direct out=) equals large @ small; returns a new tensor."""
    large = torch.randn(N, K, dtype=torch.float64)
    small = torch.randn(K, K, dtype=torch.float64)
    expected = large @ small
    result = chunked_matmul(large=large, small=small, num_divide=num_divide)
    assert result is not large, "not-inplace must return a new tensor"
    assert result.shape == (N, K), f"unexpected shape {result.shape=} vs {(N, K)=}"
    assert torch.allclose(
        result, expected
    ), f"chunked product differs from plain matmul, {num_divide=}"


@pytest.mark.parametrize("num_divide", [None, 0, 2])
def test_supports_autograd(num_divide: Optional[int]) -> None:
    """not-inplace grad path backpropagates; forward result and grads match a plain large @ small."""
    large = torch.randn(10, 5, dtype=torch.float64, requires_grad=True)
    small = torch.randn(5, 5, dtype=torch.float64, requires_grad=True)
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


@pytest.mark.parametrize("num_divide", [None, 0, 2])
def test_inplace_overwrites_large(num_divide: Optional[int]) -> None:
    """in-place path overwrites large, returns the large object, and matches a plain matmul."""
    large = torch.randn(10, 5, dtype=torch.float64)
    small = torch.randn(5, 5, dtype=torch.float64)
    expected = large.clone() @ small
    result = chunked_matmul(
        large=large, small=small, inplace=True, num_divide=num_divide
    )
    assert result is large, "inplace=True must return the large object"
    assert torch.allclose(
        large, expected
    ), f"in-place product differs from plain matmul, {num_divide=}"


def test_not_inplace_shrinks_and_resumes_on_oom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """a first-chunk CUDA OOM shrinks the chunk and resumes from the failed chunk, completing correctly."""
    large = torch.randn(20, 6, dtype=torch.float64)
    small = torch.randn(6, 6, dtype=torch.float64)
    expected = large @ small
    real_chunk = chunked_matmul_module._matmul_chunk
    state = {"calls": 0, "first_rows": None}

    def fake_chunk(
        large: torch.Tensor, small: torch.Tensor, out: torch.Tensor, direct: bool
    ) -> None:
        state["calls"] += 1
        if state["calls"] == 1:
            state["first_rows"] = large.shape[0]
            raise torch.cuda.OutOfMemoryError("simulated OOM on first chunk")
        real_chunk(large=large, small=small, out=out, direct=direct)

    monkeypatch.setattr(chunked_matmul_module, "_matmul_chunk", fake_chunk)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    result = chunked_matmul(large=large, small=small, max_divide=2)
    assert (
        state["first_rows"] == 20
    ), f"first attempt should use the full batch, got {state['first_rows']=}"
    assert state["calls"] >= 2, f"expected a retry after OOM, got {state['calls']=}"
    assert torch.allclose(
        result, expected
    ), "result after shrink-and-resume differs from plain matmul"


def test_inplace_shrinks_without_double_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """an in-place CUDA OOM resumes without re-transforming any already-written chunk."""
    large = torch.randn(20, 6, dtype=torch.float64)
    small = torch.randn(6, 6, dtype=torch.float64)
    expected = large.clone() @ small
    real_chunk = chunked_matmul_module._matmul_chunk
    state = {"calls": 0}

    def fake_chunk(
        large: torch.Tensor, small: torch.Tensor, out: torch.Tensor, direct: bool
    ) -> None:
        # OOM on the first full-batch attempt and again on a later chunk after one has
        # already been written: a correct resume continues from the failed offset, so an
        # already-written chunk is never transformed twice; a wrong restart-from-zero would.
        state["calls"] += 1
        if state["calls"] in (1, 3):
            raise torch.cuda.OutOfMemoryError("simulated OOM")
        real_chunk(large=large, small=small, out=out, direct=direct)

    monkeypatch.setattr(chunked_matmul_module, "_matmul_chunk", fake_chunk)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    result = chunked_matmul(large=large, small=small, inplace=True, max_divide=3)
    assert result is large, "inplace=True must return the large object"
    assert torch.allclose(
        large, expected
    ), "in-place resume re-transformed an already-written chunk"


def test_raises_after_max_divide_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    """OOM persisting past max_divide raises torch.cuda.OutOfMemoryError."""
    large = torch.randn(20, 6, dtype=torch.float64)
    small = torch.randn(6, 6, dtype=torch.float64)

    def always_oom(
        large: torch.Tensor, small: torch.Tensor, out: torch.Tensor, direct: bool
    ) -> None:
        raise torch.cuda.OutOfMemoryError("simulated persistent OOM")

    monkeypatch.setattr(chunked_matmul_module, "_matmul_chunk", always_oom)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        chunked_matmul(large=large, small=small, max_divide=2)


@pytest.mark.parametrize(
    "large, small",
    [
        (torch.randn(4), torch.randn(4, 4)),
        (torch.randn(2, 4, 3), torch.randn(3, 3)),
        (torch.randn(4, 3), torch.randn(3)),
        (torch.randn(4, 3), torch.randn(2, 3, 3)),
    ],
)
def test_rejects_non_2d_operands(large: torch.Tensor, small: torch.Tensor) -> None:
    """a vector, batched, or N-D large or small raises an assertion (both operands must be 2D)."""
    with pytest.raises(AssertionError):
        chunked_matmul(large=large, small=small)


def test_rejects_non_square_small() -> None:
    """a 2D but non-square small raises an assertion (small must be square)."""
    large = torch.randn(5, 4, dtype=torch.float64)
    small = torch.randn(4, 3, dtype=torch.float64)
    with pytest.raises(AssertionError):
        chunked_matmul(large=large, small=small)


def test_rejects_mismatched_inner_dim() -> None:
    """large.shape[1] != small.shape[0] raises an assertion (inner dimensions must match)."""
    large = torch.randn(5, 4, dtype=torch.float64)
    small = torch.randn(3, 3, dtype=torch.float64)
    with pytest.raises(AssertionError):
        chunked_matmul(large=large, small=small)


def test_rejects_mismatched_dtype() -> None:
    """large and small of different dtypes raise an assertion (operands must share dtype)."""
    large = torch.randn(5, 5, dtype=torch.float64)
    small = torch.randn(5, 5, dtype=torch.float32)
    with pytest.raises(AssertionError):
        chunked_matmul(large=large, small=small)


def test_inplace_rejects_grad() -> None:
    """inplace=True with a grad-requiring operand raises an assertion (in-place overwrite is illegal under autograd)."""
    large = torch.randn(10, 5, dtype=torch.float64, requires_grad=True)
    small = torch.randn(5, 5, dtype=torch.float64)
    with pytest.raises(AssertionError):
        chunked_matmul(large=large, small=small, inplace=True)
