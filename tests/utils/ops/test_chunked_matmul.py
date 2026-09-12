import pytest
import torch

from utils.ops.chunked_matmul import chunked_matmul


def test_matches_plain_matmul() -> None:
    """not-inplace no-grad path (direct out=) equals large @ small across N and num_divide splits (including the unchunked default), returning a new tensor.

    Args:
        None.

    Returns:
        None.
    """
    for N, K, num_divide in (
        (1, 3, None),
        (10, 5, 0),
        (10, 5, 1),
        (10, 5, 2),
        (37, 7, 3),
        (100, 9, 4),
    ):
        large = torch.randn(N, K, dtype=torch.float64)
        small = torch.randn(K, K, dtype=torch.float64)
        result = chunked_matmul(large=large, small=small, num_divide=num_divide)
        assert result.shape == (N, K) and torch.allclose(
            result, large @ small
        ), f"chunked product differs from plain matmul, {N=} {K=} {num_divide=} {result.shape=}"
        assert (
            result is not large
        ), f"not-inplace must return a new tensor, {N=} {K=} {num_divide=}"


def test_supports_autograd() -> None:
    """not-inplace grad path backpropagates; forward result and both grads match a plain large @ small across num_divide splits.

    Args:
        None.

    Returns:
        None.
    """
    for num_divide in (None, 0, 2):
        large = torch.randn(10, 5, dtype=torch.float64, requires_grad=True)
        small = torch.randn(5, 5, dtype=torch.float64, requires_grad=True)
        result = chunked_matmul(large=large, small=small, num_divide=num_divide)
        result.sum().backward()
        ref_large = large.detach().clone().requires_grad_(True)
        ref_small = small.detach().clone().requires_grad_(True)
        expected = ref_large @ ref_small
        expected.sum().backward()
        assert torch.allclose(
            result, expected
        ), f"chunked forward differs from plain matmul, {num_divide=}"
        assert torch.allclose(
            large.grad, ref_large.grad
        ), f"large.grad differs from plain-matmul grad, {num_divide=}"
        assert torch.allclose(
            small.grad, ref_small.grad
        ), f"small.grad differs from plain-matmul grad, {num_divide=}"


def test_inplace_overwrites_large() -> None:
    """in-place path overwrites large, returns the large object, and matches a plain matmul across num_divide splits.

    Args:
        None.

    Returns:
        None.
    """
    for num_divide in (None, 0, 2):
        large = torch.randn(10, 5, dtype=torch.float64)
        small = torch.randn(5, 5, dtype=torch.float64)
        expected = large @ small
        result = chunked_matmul(
            large=large, small=small, inplace=True, num_divide=num_divide
        )
        assert (
            result is large
        ), f"inplace=True must return the large object, {num_divide=}"
        assert torch.allclose(
            large, expected
        ), f"in-place product differs from plain matmul, {num_divide=}"


def test_not_inplace_shrinks_and_resumes_on_oom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """a first-chunk CUDA OOM shrinks the chunk and resumes from the failed chunk, completing correctly (not-inplace).

    Args:
        monkeypatch: pytest fixture that swaps the module-level _matmul_chunk for the test's duration.

    Returns:
        None.
    """
    large = torch.randn(20, 6, dtype=torch.float64)
    small = torch.randn(6, 6, dtype=torch.float64)
    # chunked_matmul's own module namespace: utils.ops rebinds the name chunked_matmul to the function, so the submodule is reached through the function's globals.
    real_chunk = chunked_matmul.__globals__["_matmul_chunk"]
    state = {"calls": 0}

    def fake_chunk(
        large: torch.Tensor, small: torch.Tensor, out: torch.Tensor, direct: bool
    ) -> None:
        """Raise a CUDA OOM on the first call only, and write the real chunk product otherwise.

        Args:
            large: Left operand chunk of shape [b, K].
            small: Right square operand of shape [K, K].
            out: Destination chunk of shape [b, K].
            direct: Whether the real chunk writer writes straight into out.

        Returns:
            None.
        """
        state["calls"] += 1
        if state["calls"] == 1:
            raise torch.cuda.OutOfMemoryError("simulated OOM on first chunk")
        real_chunk(large=large, small=small, out=out, direct=direct)

    monkeypatch.setitem(chunked_matmul.__globals__, "_matmul_chunk", fake_chunk)
    result = chunked_matmul(large=large, small=small, max_divide=2)
    assert torch.allclose(
        result, large @ small
    ), "result after shrink-and-resume differs from plain matmul"
    assert state["calls"] >= 2, f"expected a retry after OOM, got {state['calls']=}"


def test_inplace_shrinks_without_double_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """an in-place first-chunk CUDA OOM resumes without re-transforming any already-written chunk, so large equals a single plain matmul.

    Args:
        monkeypatch: pytest fixture that swaps the module-level _matmul_chunk for the test's duration.

    Returns:
        None.
    """
    large = torch.randn(20, 6, dtype=torch.float64)
    small = torch.randn(6, 6, dtype=torch.float64)
    expected = large @ small
    real_chunk = chunked_matmul.__globals__["_matmul_chunk"]
    state = {"calls": 0}

    def fake_chunk(
        large: torch.Tensor, small: torch.Tensor, out: torch.Tensor, direct: bool
    ) -> None:
        """Raise a CUDA OOM on the first call and on the third, after a chunk has been written, and write the real chunk product otherwise.

        Args:
            large: Left operand chunk of shape [b, K].
            small: Right square operand of shape [K, K].
            out: Destination chunk of shape [b, K], aliasing large's rows.
            direct: Whether the real chunk writer writes straight into out.

        Returns:
            None.
        """
        # A correct resume continues from the failed offset, so an already-written chunk is never transformed twice; a wrong restart-from-zero would.
        state["calls"] += 1
        if state["calls"] in (1, 3):
            raise torch.cuda.OutOfMemoryError("simulated OOM")
        real_chunk(large=large, small=small, out=out, direct=direct)

    monkeypatch.setitem(chunked_matmul.__globals__, "_matmul_chunk", fake_chunk)
    chunked_matmul(large=large, small=small, inplace=True, max_divide=3)
    assert torch.allclose(
        large, expected
    ), "in-place resume re-transformed an already-written chunk"


def test_raises_after_max_divide_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    """OOM persisting past max_divide raises torch.cuda.OutOfMemoryError.

    Args:
        monkeypatch: pytest fixture that swaps the module-level _matmul_chunk for the test's duration.

    Returns:
        None.
    """
    large = torch.randn(20, 6, dtype=torch.float64)
    small = torch.randn(6, 6, dtype=torch.float64)

    def always_oom(
        large: torch.Tensor, small: torch.Tensor, out: torch.Tensor, direct: bool
    ) -> None:
        """Raise a CUDA OOM on every call.

        Args:
            large: Left operand chunk of shape [b, K].
            small: Right square operand of shape [K, K].
            out: Destination chunk of shape [b, K].
            direct: Whether the real chunk writer would write straight into out.

        Returns:
            None.
        """
        raise torch.cuda.OutOfMemoryError("simulated persistent OOM")

    monkeypatch.setitem(chunked_matmul.__globals__, "_matmul_chunk", always_oom)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        chunked_matmul(large=large, small=small, max_divide=2)


def test_rejects_non_2d_large() -> None:
    """a 1D or N-D large raises an assertion (the chunked operand must be 2D), as does a 1D small.

    Args:
        None.

    Returns:
        None.
    """
    for large, small in (
        (torch.randn(4), torch.randn(4, 4)),
        (torch.randn(2, 4, 3), torch.randn(3, 3)),
        (torch.randn(4, 3), torch.randn(3)),
    ):
        with pytest.raises(AssertionError):
            chunked_matmul(large=large, small=small)


def test_batched_small_broadcasts_onto_the_product() -> None:
    """a [B, K, K] small gives a [B, N, K] product whose every slice equals that slice's own small multiplied alone under the same split, bit for bit on cpu and within floating-point rounding on cuda.

    Args:
        None.

    Returns:
        None.
    """
    B, K = 3, 4
    devices = [torch.device("cpu")] + (
        [torch.device("cuda")] if torch.cuda.is_available() else []
    )
    for device in devices:
        # The small row counts where CUDA's batched and unbatched products disagree in the last place.
        for N in (1, 17, 25, 33, 100):
            large = torch.randn(N, K, dtype=torch.float32, device=device)
            small = torch.randn(B, K, K, dtype=torch.float32, device=device)
            for num_divide in (None, 0, 2, 3):
                result = chunked_matmul(large=large, small=small, num_divide=num_divide)
                for b in range(B):
                    one_small_result = chunked_matmul(
                        large=large, small=small[b], num_divide=num_divide
                    )
                    if device.type == "cpu":
                        # The same split on both sides, since a row chunk may round unlike the whole product.
                        assert torch.equal(
                            result[b], one_small_result
                        ), f"slice {b=} differs from its own small multiplied alone, {device=} {N=} {num_divide=}"
                    else:
                        # CUDA picks a batched kernel for several entries, which rounds unlike the single product at some row counts; the tolerances are float32 rounding (torch.testing's float32 defaults).
                        assert torch.allclose(
                            result[b], one_small_result, rtol=1.3e-6, atol=1e-5
                        ), f"slice {b=} disagrees with its own small multiplied alone beyond float32 rounding, {device=} {N=} {num_divide=} {(result[b] - one_small_result).abs().max()=}"


def test_inplace_rejects_batched_small() -> None:
    """inplace=True with a batched small raises an assertion (the product is wider than large, leaving nothing to overwrite in place)."""
    large = torch.randn(10, 5, dtype=torch.float64)
    small = torch.randn(3, 5, 5, dtype=torch.float64)
    with pytest.raises(AssertionError):
        chunked_matmul(large=large, small=small, inplace=True)


def test_rejects_non_square_small() -> None:
    """a non-square small raises an assertion (small must be square in its trailing two axes)."""
    large = torch.randn(5, 4, dtype=torch.float64)
    small = torch.randn(3, 4, 3, dtype=torch.float64)
    with pytest.raises(AssertionError):
        chunked_matmul(large=large, small=small)


def test_rejects_mismatched_inner_dim() -> None:
    """large.shape[1] != small.shape[-2] raises an assertion (inner dimensions must match)."""
    large = torch.randn(5, 4, dtype=torch.float64)
    small = torch.randn(4, 3, 3, dtype=torch.float64)
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
