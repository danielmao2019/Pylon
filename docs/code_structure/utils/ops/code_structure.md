# utils/ops — code structure

`utils/ops/chunked_matmul.py`

```text
chunked_matmul.py
├── import math
├── import torch
├── from typing import Optional
├── def chunked_matmul(large: torch.Tensor, small: torch.Tensor, max_divide: int = 0, num_divide: Optional[int] = None) -> torch.Tensor
│   ├── # Multiplies a large 2D tensor by a small 2D tensor on its right, splitting the large's first dim into a fixed num_divide batches, else progressively halving the batch on CUDA OOM up to max_divide.
│   ├── impls N = large.shape[0]
│   ├── if num_divide is not None
│   │   ├── impls bs = max(1, math.ceil(N / 2 ** num_divide))
│   │   ├── calls _chunked_matmul_batched
│   │   └── return  # the [N, M] product
│   ├── impls n = 0
│   ├── while n <= max_divide
│   │   ├── impls bs = max(1, math.ceil(N / 2 ** n))
│   │   ├── try
│   │   │   ├── calls _chunked_matmul_batched
│   │   │   └── return  # the [N, M] product
│   │   ├── except torch.cuda.OutOfMemoryError
│   │   │   ├── impls increment n
│   │   │   ├── impls release cached CUDA memory
│   │   │   └── continue
│   │   └── except Exception
│   │       └── raise
│   └── raise torch.cuda.OutOfMemoryError  # exhausted max_divide halvings
└── def _chunked_matmul_batched(large: torch.Tensor, small: torch.Tensor, batch_size: int) -> torch.Tensor
    ├── # Multiplies large by small in row-chunks of batch_size.
    ├── impls allocate the [N, M] result on large's device and dtype  # impls-node-one-step:skip
    ├── for each row start i in range(0, N, batch_size)
    │   └── impls result[i : i + batch_size] = large[i : i + batch_size] @ small
    └── return  # the [N, M] product
```
