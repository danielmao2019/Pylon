# utils/ops — tests structure

`tests/utils/ops/test_chunked_matmul.py`

```text
test_chunked_matmul.py
├── def test_matches_plain_matmul
│   └── # chunked_matmul equals large @ small across several N and num_divide splits.
├── def test_supports_autograd
│   └── # chunked_matmul backpropagates and its grads match a plain large @ small across num_divide splits.
├── def test_halves_batch_on_cuda_oom
│   └── # a first-batch CUDA OOM triggers a halving retry that then completes correctly.
├── def test_raises_after_max_divide_exhausted
│   └── # OOM persisting past max_divide raises torch.cuda.OutOfMemoryError.
└── def test_rejects_non_2d_operands
    └── # a vector, batched, or N-D large or small raises an assertion (both operands must be 2D).
```
