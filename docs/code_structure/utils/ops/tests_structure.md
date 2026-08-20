# utils/ops — tests structure

`tests/utils/ops/test_chunked_matmul.py`

```text
test_chunked_matmul.py
├── import pytest
├── import torch
├── from utils.ops.chunked_matmul import chunked_matmul
├── def test_matches_plain_matmul
│   ├── # not-inplace no-grad path (direct out=) equals large @ small across N and num_divide splits (including the unchunked default), returning a new tensor.
│   ├── for each (N, num_divide) pair over several row counts and splits, num_divide left unset among them
│   │   ├── calls chunked_matmul(large=large, small=small, num_divide=num_divide)
│   │   ├── impls assert the result equals large @ small
│   │   └── impls assert the result is a new tensor, not large itself
│   └── return
├── def test_supports_autograd
│   ├── # not-inplace grad path backpropagates; forward result and both grads match a plain large @ small across num_divide splits.
│   ├── for each num_divide split
│   │   ├── calls chunked_matmul(large=grad-requiring large, small=grad-requiring small, num_divide=num_divide)
│   │   ├── impls backpropagate the result's sum
│   │   ├── impls assert the result equals large @ small
│   │   ├── impls assert large's grad equals the plain matmul's own grad on large
│   │   └── impls assert small's grad equals the plain matmul's own grad on small
│   └── return
├── def test_inplace_overwrites_large
│   ├── # in-place path overwrites large, returns the large object, and matches a plain matmul across num_divide splits.
│   ├── for each num_divide split
│   │   ├── impls expected = large @ small, taken before the call overwrites large
│   │   ├── calls chunked_matmul(large=large, small=small, inplace=True, num_divide=num_divide)
│   │   ├── impls assert the returned object is large itself
│   │   └── impls assert large equals expected
│   └── return
├── def test_not_inplace_shrinks_and_resumes_on_oom
│   ├── # a first-chunk CUDA OOM shrinks the chunk and resumes from the failed chunk, completing correctly (not-inplace).
│   ├── impls patch _matmul_chunk to raise torch.cuda.OutOfMemoryError on its first call only
│   ├── calls chunked_matmul(large=large, small=small, max_divide=1)
│   ├── impls assert the result equals large @ small
│   ├── impls assert the patched chunk writer was re-entered at the failed chunk's own row offset
│   └── return
├── def test_inplace_shrinks_without_double_transform
│   ├── # an in-place first-chunk CUDA OOM resumes without re-transforming any already-written chunk, so large equals a single plain matmul.
│   ├── impls expected = large @ small, taken before the call overwrites large
│   ├── impls patch _matmul_chunk to raise torch.cuda.OutOfMemoryError once, after it has written its first chunk
│   ├── calls chunked_matmul(large=large, small=small, inplace=True, max_divide=1)
│   ├── impls assert large equals expected
│   └── return
├── def test_raises_after_max_divide_exhausted
│   ├── # OOM persisting past max_divide raises torch.cuda.OutOfMemoryError.
│   ├── impls patch _matmul_chunk to raise torch.cuda.OutOfMemoryError on every call
│   ├── with pytest.raises(torch.cuda.OutOfMemoryError)
│   │   └── calls chunked_matmul(large=large, small=small, max_divide=2)
│   └── return
├── def test_rejects_non_2d_operands
│   ├── # a vector, batched, or N-D large or small raises an assertion (both operands must be 2D).
│   ├── for each of a 1D, a 3D and a 4D operand, taken in turn as large and as small
│   │   └── with pytest.raises(AssertionError)
│   │       └── calls chunked_matmul(large=large, small=small)
│   └── return
├── def test_rejects_non_square_small
│   ├── # a 2D but non-square small raises an assertion (small must be square).
│   ├── with pytest.raises(AssertionError)
│   │   └── calls chunked_matmul(large=large, small=a 2D non-square small)
│   └── return
├── def test_rejects_mismatched_inner_dim
│   ├── # large.shape[1] != small.shape[0] raises an assertion (inner dimensions must match).
│   ├── with pytest.raises(AssertionError)
│   │   └── calls chunked_matmul(large=large, small=a square small whose side differs from large's inner dim)
│   └── return
├── def test_rejects_mismatched_dtype
│   ├── # large and small of different dtypes raise an assertion (operands must share dtype).
│   ├── with pytest.raises(AssertionError)
│   │   └── calls chunked_matmul(large=a float32 large, small=a float64 small)
│   └── return
└── def test_inplace_rejects_grad
    ├── # inplace=True with a grad-requiring operand raises an assertion (in-place overwrite is illegal under autograd).
    ├── for each of a grad-requiring large and a grad-requiring small
    │   └── with pytest.raises(AssertionError)
    │       └── calls chunked_matmul(large=large, small=small, inplace=True)
    └── return
```
