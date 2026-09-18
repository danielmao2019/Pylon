# utils/ops: tests structure

`tests/utils/ops/test_chunked_matmul.py`

```text
test_chunked_matmul.py
├── import pytest
├── import torch
├── from utils.ops.chunked_matmul import chunked_matmul
├── def test_matches_plain_matmul
│   ├── # not-inplace no-grad path (direct out=) equals large @ small across N and num_divide splits (including the unchunked default), returning a new tensor.
│   ├── for each (N, K, num_divide) triple over several row counts, inner dims and splits, num_divide left unset among them
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
│   ├── calls chunked_matmul(large=large, small=small, max_divide=2)
│   ├── impls assert the result equals large @ small
│   ├── impls assert the patched chunk writer was re-entered after the OOM
│   └── return
├── def test_inplace_shrinks_without_double_transform
│   ├── # an in-place first-chunk CUDA OOM resumes without re-transforming any already-written chunk, so large equals a single plain matmul.
│   ├── impls expected = large @ small, taken before the call overwrites large
│   ├── impls patch _matmul_chunk to raise torch.cuda.OutOfMemoryError on its first call, plus once more after a chunk has been written
│   ├── calls chunked_matmul(large=large, small=small, inplace=True, max_divide=3)
│   ├── impls assert large equals expected
│   └── return
├── def test_raises_after_max_divide_exhausted
│   ├── # OOM persisting past max_divide raises torch.cuda.OutOfMemoryError.
│   ├── impls patch _matmul_chunk to raise torch.cuda.OutOfMemoryError on every call
│   ├── with pytest.raises(torch.cuda.OutOfMemoryError)
│   │   └── calls chunked_matmul(large=large, small=small, max_divide=2)
│   └── return
├── def test_rejects_non_2d_large
│   ├── # a 1D or N-D large raises an assertion (the chunked operand must be 2D), as does a 1D small.
│   └── for large, small in ((a 1D [4] large, a [4, 4] small), (a 3D [2, 4, 3] large, a [3, 3] small), (a [4, 3] large, a 1D [3] small))  # standard-normal draws
│       └── with pytest.raises(AssertionError)
│           └── calls chunked_matmul(large=large, small=small)
├── def test_batched_small_broadcasts_onto_the_product
│   ├── # a [B, K, K] small gives a [B, N, K] product whose every slice equals that slice's own small multiplied alone under the same split, bit for bit on cpu and within floating-point rounding on cuda.
│   ├── impls B, K = 3, 4
│   ├── impls devices = a list holding the cpu device
│   ├── if cuda is available
│   │   └── impls append the cuda device to devices
│   └── for device in devices
│       └── for N in (1, 17, 25, 33, 100)  # the small row counts where CUDA's batched and unbatched products disagree in the last place
│           ├── impls large = a float32 [N, K] standard-normal tensor on device
│           ├── impls small = a float32 [B, K, K] standard-normal tensor on device
│           └── for num_divide in (None, 0, 2, 3)  # several splits, the unchunked default among them
│               ├── calls chunked_matmul(large=large, small=small, num_divide=num_divide)  # -> result
│               └── for b in range(B)
│                   ├── calls chunked_matmul(large=large, small=small[b], num_divide=num_divide)  # -> one_small_result
│                   ├── if device.type == 'cpu'
│                   │   └── assert torch.equal(result[b], one_small_result)  # the same split on both sides, since a row chunk may round unlike the whole product
│                   └── else
│                       └── assert torch.allclose(result[b], one_small_result, rtol=1.3e-6, atol=1e-5)  # CUDA picks a batched kernel for several entries, which rounds unlike the single product at some row counts, and the tolerances are torch.testing's float32 defaults
├── def test_inplace_rejects_batched_small
│   ├── # inplace=True with a batched small raises an assertion (the product is wider than large, leaving nothing to overwrite in place).
│   ├── impls large = a float64 [10, 5] standard-normal tensor
│   ├── impls small = a float64 [3, 5, 5] standard-normal tensor  # a batched small
│   └── with pytest.raises(AssertionError)
│       └── calls chunked_matmul(large=large, small=small, inplace=True)
├── def test_rejects_non_square_small
│   ├── # a non-square small raises an assertion (small must be square in its trailing two axes).
│   ├── impls large = a float64 [5, 4] standard-normal tensor
│   ├── impls small = a float64 [3, 4, 3] standard-normal tensor  # its trailing two axes differ
│   └── with pytest.raises(AssertionError)
│       └── calls chunked_matmul(large=large, small=small)
├── def test_rejects_mismatched_inner_dim
│   ├── # large.shape[1] != small.shape[-2] raises an assertion (inner dimensions must match).
│   ├── impls large = a float64 [5, 4] standard-normal tensor
│   ├── impls small = a float64 [4, 3, 3] standard-normal tensor  # a square small whose side of 3 differs from large's inner dim of 4
│   └── with pytest.raises(AssertionError)
│       └── calls chunked_matmul(large=large, small=small)
├── def test_rejects_mismatched_dtype
│   ├── # large and small of different dtypes raise an assertion (operands must share dtype).
│   ├── with pytest.raises(AssertionError)
│   │   └── calls chunked_matmul(large=a float64 large, small=a float32 small)
│   └── return
└── def test_inplace_rejects_grad
    ├── # inplace=True with a grad-requiring operand raises an assertion (in-place overwrite is illegal under autograd).
    ├── with pytest.raises(AssertionError)
    │   └── calls chunked_matmul(large=a grad-requiring large, small=small, inplace=True)
    └── return
```
