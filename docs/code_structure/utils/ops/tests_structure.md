# utils/ops — tests structure

`tests/utils/ops/test_chunked_matmul.py`

```text
test_chunked_matmul.py
├── def test_matches_plain_matmul
│   └── # chunked_matmul equals large @ small across several N and num_divide splits (square small).
├── def test_supports_autograd
│   └── # the not-inplace grad path backpropagates and its grads match a plain large @ small across num_divide splits.
├── def test_inplace_overwrites_large
│   └── # inplace=True returns the large object overwritten with large @ small, matching a plain matmul across num_divide splits.
├── def test_inplace_rejects_grad
│   └── # inplace=True with a grad-requiring operand raises an assertion (in-place overwrite is illegal under autograd).
├── def test_halves_batch_on_cuda_oom
│   └── # a first-chunk CUDA OOM triggers a shrink-and-resume that completes correctly without recomputing the done chunks.
├── def test_raises_after_max_divide_exhausted
│   └── # OOM persisting past max_divide raises torch.cuda.OutOfMemoryError.
├── def test_rejects_non_square_small
│   └── # a 2D but non-square small raises an assertion (small must be square).
└── def test_rejects_non_2d_operands
    └── # a vector, batched, or N-D large or small raises an assertion (both operands must be 2D).
```
