# utils/ops — folder structure

## Code folder structure

```text
utils/ops/
├── __init__.py
├── apply.py
├── chunked_matmul.py        # large×small chunked matmul, shrinking the chunk on CUDA OOM until it fits
├── dict_as_tensor.py
├── dict_ops.py
└── materialize_tensor.py
```

## Tests folder structure

```text
tests/utils/ops/
└── test_chunked_matmul.py   # chunked-matmul numeric equality vs a plain matmul + the OOM shrink-retry behavior
```
