# utils/ops — tests structure

## Tests implementation structure

`tests/utils/ops/test_chunked_matmul.py`

```text
test_chunked_matmul.py
├── import importlib
├── from typing import Optional
├── import pytest
├── import torch
├── from utils.ops.chunked_matmul import chunked_matmul
├── chunked_matmul_module  # importlib.import_module("utils.ops.chunked_matmul"), reaching the true module because the package __init__ rebinds the name and shadows the submodule; every _matmul_chunk monkeypatch targets it
├── @pytest.mark.parametrize def test_matches_plain_matmul(N: int, K: int, num_divide: Optional[int]) -> None  # over the (1, 3, None), (10, 5, 0), (10, 5, 1), (10, 5, 2), (37, 7, 3) and (100, 9, 4) N / K / num_divide triples
│   ├── # not-inplace no-grad path (direct out=) equals large @ small across N and num_divide splits (including the unchunked default), returning a new tensor.
│   ├── impls large — a float64 [N, K] random tensor
│   ├── impls small — a float64 [K, K] random tensor
│   ├── impls expected — large @ small
│   ├── calls chunked_matmul(large=large, small=small, num_divide=num_divide)
│   ├── assert the result is not the large object
│   ├── assert the result has shape (N, K)
│   └── assert the result is all-close to expected
├── @pytest.mark.parametrize def test_supports_autograd(num_divide: Optional[int]) -> None  # over the None, 0 and 2 num_divide values
│   ├── # not-inplace grad path backpropagates; forward result and both grads match a plain large @ small across num_divide splits.
│   ├── impls large — a grad-requiring float64 [10, 5] random tensor
│   ├── impls small — a grad-requiring float64 [5, 5] random tensor
│   ├── impls ref_large — a detached grad-requiring clone of large
│   ├── impls ref_small — a detached grad-requiring clone of small
│   ├── calls chunked_matmul(large=large, small=small, num_divide=num_divide)
│   ├── impls expected — ref_large @ ref_small
│   ├── assert the result is all-close to expected
│   ├── impls backpropagate the result's sum
│   ├── impls backpropagate expected's sum
│   ├── assert large.grad is not None
│   ├── assert small.grad is not None
│   ├── assert large.grad is all-close to ref_large.grad
│   └── assert small.grad is all-close to ref_small.grad
├── @pytest.mark.parametrize def test_inplace_overwrites_large(num_divide: Optional[int]) -> None  # over the None, 0 and 2 num_divide values
│   ├── # in-place path overwrites large, returns the large object, and matches a plain matmul across num_divide splits.
│   ├── impls large — a float64 [10, 5] random tensor
│   ├── impls small — a float64 [5, 5] random tensor
│   ├── impls expected — a clone of large times small, taken before the call overwrites large
│   ├── calls chunked_matmul(large=large, small=small, inplace=True, num_divide=num_divide)
│   ├── assert the returned object is large itself
│   └── assert large is all-close to expected
├── def test_not_inplace_shrinks_and_resumes_on_oom(monkeypatch: pytest.MonkeyPatch) -> None
│   ├── # a first-chunk CUDA OOM shrinks the chunk and resumes from the failed chunk, completing correctly (not-inplace).
│   ├── impls large — a float64 [20, 6] random tensor
│   ├── impls small — a float64 [6, 6] random tensor
│   ├── impls expected — large @ small
│   ├── impls real_chunk — the module's own _matmul_chunk, captured before it is patched
│   ├── impls state — a dict carrying a call counter and the first attempt's row count  # impls-node-one-step:skip
│   ├── def fake_chunk(large: torch.Tensor, small: torch.Tensor, out: torch.Tensor, direct: bool) -> None [local]
│   │   ├── # Fails the very first chunk with a CUDA OOM and records the row count it was handed, so the test can tell how wide that attempt was.
│   │   ├── impls increment state's call counter
│   │   ├── if this is the first call
│   │   │   ├── impls record this chunk's row count as state's first_rows
│   │   │   └── raise torch.cuda.OutOfMemoryError  # a simulated OOM on the first chunk
│   │   └── impls real_chunk(large=large, small=small, out=out, direct=direct)  # real_chunk is the pre-patch _matmul_chunk this local captured
│   ├── calls monkeypatch.setattr(chunked_matmul_module, "_matmul_chunk", fake_chunk)
│   ├── calls monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
│   ├── calls chunked_matmul(large=large, small=small, max_divide=2)
│   ├── assert the recorded first_rows equals 20, so the first attempt used the full batch
│   ├── assert the chunk writer was entered at least twice, so a retry followed the OOM
│   └── assert the result is all-close to expected
├── def test_inplace_shrinks_without_double_transform(monkeypatch: pytest.MonkeyPatch) -> None
│   ├── # an in-place first-chunk CUDA OOM resumes without re-transforming any already-written chunk, so large equals a single plain matmul.
│   ├── impls large — a float64 [20, 6] random tensor
│   ├── impls small — a float64 [6, 6] random tensor
│   ├── impls expected — a clone of large times small, taken before the call overwrites large
│   ├── impls real_chunk — the module's own _matmul_chunk, captured before it is patched
│   ├── impls state — a dict carrying a call counter
│   ├── def fake_chunk(large: torch.Tensor, small: torch.Tensor, out: torch.Tensor, direct: bool) -> None [local]
│   │   ├── # Fails on calls 1 and 3 — the full-batch attempt and a later chunk after one has already been written — so a restart-from-zero resume would transform that written chunk twice.
│   │   ├── impls increment state's call counter
│   │   ├── if the call counter is 1 or 3
│   │   │   └── raise torch.cuda.OutOfMemoryError  # a simulated OOM
│   │   └── impls real_chunk(large=large, small=small, out=out, direct=direct)  # real_chunk is the pre-patch _matmul_chunk this local captured
│   ├── calls monkeypatch.setattr(chunked_matmul_module, "_matmul_chunk", fake_chunk)
│   ├── calls monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
│   ├── calls chunked_matmul(large=large, small=small, inplace=True, max_divide=3)
│   ├── assert the returned object is large itself
│   └── assert large is all-close to expected
├── def test_raises_after_max_divide_exhausted(monkeypatch: pytest.MonkeyPatch) -> None
│   ├── # OOM persisting past max_divide raises torch.cuda.OutOfMemoryError.
│   ├── impls large — a float64 [20, 6] random tensor
│   ├── impls small — a float64 [6, 6] random tensor
│   ├── def always_oom(large: torch.Tensor, small: torch.Tensor, out: torch.Tensor, direct: bool) -> None [local]
│   │   ├── # Fails every chunk, so none of the shrinks the op attempts can succeed.
│   │   └── raise torch.cuda.OutOfMemoryError  # a simulated persistent OOM
│   ├── calls monkeypatch.setattr(chunked_matmul_module, "_matmul_chunk", always_oom)
│   ├── calls monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
│   └── with pytest.raises(torch.cuda.OutOfMemoryError)
│       └── calls chunked_matmul(large=large, small=small, max_divide=2)
├── @pytest.mark.parametrize def test_rejects_non_2d_operands(large: torch.Tensor, small: torch.Tensor) -> None  # over four (large, small) pairs — a 1-D large, a 3-D large, a 1-D small, and a 3-D small
│   ├── # a vector, batched, or N-D large or small raises an assertion (both operands must be 2D).
│   └── with pytest.raises(AssertionError)
│       └── calls chunked_matmul(large=large, small=small)
├── def test_rejects_non_square_small() -> None
│   ├── # a 2D but non-square small raises an assertion (small must be square).
│   ├── impls large — a float64 [5, 4] random tensor
│   ├── impls small — a float64 [4, 3] random tensor
│   └── with pytest.raises(AssertionError)
│       └── calls chunked_matmul(large=large, small=small)
├── def test_rejects_mismatched_inner_dim() -> None
│   ├── # large.shape[1] != small.shape[0] raises an assertion (inner dimensions must match).
│   ├── impls large — a float64 [5, 4] random tensor
│   ├── impls small — a float64 [3, 3] random tensor
│   └── with pytest.raises(AssertionError)
│       └── calls chunked_matmul(large=large, small=small)
├── def test_rejects_mismatched_dtype() -> None
│   ├── # large and small of different dtypes raise an assertion (operands must share dtype).
│   ├── impls large — a float64 [5, 5] random tensor
│   ├── impls small — a float32 [5, 5] random tensor
│   └── with pytest.raises(AssertionError)
│       └── calls chunked_matmul(large=large, small=small)
└── def test_inplace_rejects_grad() -> None
    ├── # inplace=True with a grad-requiring operand raises an assertion (in-place overwrite is illegal under autograd).
    ├── impls large — a grad-requiring float64 [10, 5] random tensor
    ├── impls small — a plain float64 [5, 5] random tensor
    └── with pytest.raises(AssertionError)
        └── calls chunked_matmul(large=large, small=small, inplace=True)
```

`tests/utils/ops/test_dict_as_tensor.py`

```text
test_dict_as_tensor.py
├── from typing import List, Dict, Union, Any, Tuple
├── import pytest
├── import numpy
├── import torch
├── from utils.ops.dict_as_tensor import buffer_allclose
├── from utils.ops.dict_as_tensor import transpose_buffer, buffer_permute
├── from utils.ops.dict_as_tensor import buffer_add
├── from utils.ops.dict_as_tensor import buffer_sub
├── from utils.ops.dict_as_tensor import buffer_mul
├── from utils.ops.dict_as_tensor import buffer_div
├── from utils.ops.dict_as_tensor import buffer_mean
├── # Each dict_as_tensor import above sits mid-file in the source, under a banner separator immediately ahead of the tests that use it.
├── @pytest.mark.parametrize def test_buffer_allclose(buffer1, buffer2, expected) -> None  # over one (buffer1, buffer2, expected) triple of two aggregated / per_datapoint score buffers, expected True
│   ├── # Two score buffers whose entries differ by under a tenth in relative terms still compare equal, so run-to-run numeric drift does not fail a comparison.
│   ├── calls buffer_allclose(buffer1, buffer2, rtol=1e-01, atol=0)
│   └── assert the result equals expected
├── @pytest.mark.parametrize def test_transpose_buffer(buffer: List[Dict[str, Any]], expected: Dict[str, List[Any]]) -> None  # over three (buffer, expected) pairs — list-of-lists, list-of-dicts, and list-of-nested-dicts
│   ├── # Transposing a two-axis buffer swaps its outer list against its inner list or dict keys, whatever the leaf payload is.
│   ├── calls transpose_buffer(buffer=buffer)
│   └── assert the result equals expected
├── @pytest.mark.parametrize def test_transpose_buffer_invalid_cases(buffer: List[Dict[str, Any]]) -> None  # over the single empty-buffer case
│   ├── # A buffer with fewer than two axes has nothing to transpose and trips the guarding assertion.
│   └── with pytest.raises(AssertionError)  # matching the "Transpose is not supported for buffers with less than 2 axes." message
│       └── calls transpose_buffer(buffer=buffer)
├── @pytest.mark.parametrize def test_buffer_permute(buffer: Any, axes: Tuple[int, ...], expected: Any) -> None  # over seven (buffer, axes, expected) triples spanning list, dict, tuple and nested payloads, plus the axes=None case
│   ├── # An explicit axes permutation reorders a buffer's axes across every container type, and axes=None reverses them.
│   ├── calls buffer_permute(buffer, axes)
│   └── assert the result equals expected
├── @pytest.mark.parametrize def test_buffer_permute_edge_cases(buffer: Any, axes: Tuple[int, ...], expected: Any) -> None  # over two (buffer, axes, expected) triples — an empty buffer, and an identity permutation
│   ├── # An empty buffer and an identity permutation both come back unchanged.
│   ├── calls buffer_permute(buffer, axes)
│   └── assert the result equals expected
├── @pytest.mark.parametrize def test_buffer_permute_invalid_axes(buffer: Any, axes: Tuple[int, ...]) -> None  # over three (buffer, axes) pairs — wrong axes length, out-of-range axis, duplicate axis
│   ├── # Axes of the wrong length, out of range, or duplicated trip the guarding assertion.
│   └── with pytest.raises(AssertionError)
│       └── calls buffer_permute(buffer, axes)
├── @pytest.mark.parametrize def test_buffer_add(buffers, expected) -> None  # over one (buffers, expected) pair of two nested single-leaf buffers
│   ├── # Adding buffers sums their leaves key by key, descending through the nesting.
│   ├── calls buffer_add(*the buffers)
│   └── assert the result equals expected
├── @pytest.mark.parametrize def test_buffer_sub(buffer, other, expected) -> None  # over one (buffer, other, expected) triple of nested single-leaf buffers
│   ├── # Subtracting one buffer from another differences their leaves key by key.
│   ├── calls buffer_sub(buffer, other)
│   └── assert the result equals expected
├── @pytest.mark.parametrize def test_buffer_mul(buffer, other, expected) -> None  # over one (buffer, other, expected) triple of nested single-leaf buffers
│   ├── # Multiplying two buffers takes the product of their leaves key by key.
│   ├── calls buffer_mul(buffer, other)
│   └── assert the result equals expected
├── @pytest.mark.parametrize def test_buffer_div(buffer, other, expected) -> None  # over one (buffer, other, expected) triple of nested single-leaf buffers
│   ├── # Dividing one buffer by another takes the quotient of their leaves key by key, yielding floats.
│   ├── calls buffer_div(buffer, other)
│   └── assert the result equals expected
└── @pytest.mark.parametrize def test_buffer_mean(buffer: List[Dict[str, Union[int, float, numpy.ndarray, torch.Tensor]]], expected: Dict[str, Union[int, float, numpy.ndarray, torch.Tensor]]) -> None  # over nine (buffer, expected) pairs covering numpy, torch, list, dict, int, float and mixed leaves
    ├── # Averaging a list of buffers reduces them leaf by leaf while keeping each leaf's own container type.
    ├── calls buffer_mean(buffer=buffer)  # -> produced
    ├── assert produced's keys equal expected's keys
    └── for key in produced's keys
        ├── assert produced[key] and expected[key] have the same type
        ├── if that type is numpy.ndarray
        │   └── assert the two arrays are element-wise equal
        ├── elif that type is torch.Tensor
        │   └── assert the two tensors are element-wise equal
        └── else
            └── assert produced[key] equals expected[key]
```
