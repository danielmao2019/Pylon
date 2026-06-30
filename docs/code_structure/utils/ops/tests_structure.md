# utils/ops — tests structure

`tests/utils/ops/test_chunked_matmul.py`

```text
test_chunked_matmul.py
├── def test_matches_plain_matmul
│   └── # not-inplace no-grad path (direct out=) equals large @ small across N and num_divide splits (including the unchunked default), returning a new tensor.
├── def test_supports_autograd
│   └── # not-inplace grad path backpropagates; forward result and both grads match a plain large @ small across num_divide splits.
├── def test_inplace_overwrites_large
│   └── # in-place path overwrites large, returns the large object, and matches a plain matmul across num_divide splits.
├── def test_not_inplace_shrinks_and_resumes_on_oom
│   └── # a first-chunk CUDA OOM shrinks the chunk and resumes from the failed chunk, completing correctly (not-inplace).
├── def test_inplace_shrinks_without_double_transform
│   └── # an in-place first-chunk CUDA OOM resumes without re-transforming any already-written chunk, so large equals a single plain matmul.
├── def test_raises_after_max_divide_exhausted
│   └── # OOM persisting past max_divide raises torch.cuda.OutOfMemoryError.
├── def test_rejects_non_2d_operands
│   └── # a vector, batched, or N-D large or small raises an assertion (both operands must be 2D).
├── def test_rejects_non_square_small
│   └── # a 2D but non-square small raises an assertion (small must be square).
├── def test_rejects_mismatched_inner_dim
│   └── # large.shape[1] != small.shape[0] raises an assertion (inner dimensions must match).
├── def test_rejects_mismatched_dtype
│   └── # large and small of different dtypes raise an assertion (operands must share dtype).
└── def test_inplace_rejects_grad
    └── # inplace=True with a grad-requiring operand raises an assertion (in-place overwrite is illegal under autograd).
```
