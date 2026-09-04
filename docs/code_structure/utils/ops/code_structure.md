# utils/ops — code structure

`utils/ops/chunked_matmul.py`

```text
chunked_matmul.py
├── import math
├── import torch
├── from typing import Optional
├── def chunked_matmul(large: torch.Tensor, small: torch.Tensor, inplace: bool = False, max_divide: int = 0, num_divide: Optional[int] = None) -> torch.Tensor
│   ├── # Multiplies a large 2D tensor by a small square 2D tensor on its right, chunking the large's first dim; resume-safe so a CUDA-OOM chunk shrink never recomputes a completed chunk; inplace overwrites large (requires no grad).
│   ├── calls _validate_inputs
│   ├── impls small = small.contiguous()
│   ├── impls N = large.shape[0]
│   ├── impls M = small.shape[1]
│   ├── impls out = large if inplace else a fresh empty [N, M] on large's device and dtype  # impls-node-one-step:skip
│   ├── impls direct = not inplace and not large.requires_grad and not small.requires_grad  # impls-node-one-step:skip
│   ├── impls bs = max(1, math.ceil(N / 2 ** num_divide)) if num_divide is not None else N  # impls-node-one-step:skip
│   ├── impls i = 0
│   ├── impls divides = 0
│   ├── while i < N
│   │   ├── impls j = min(N, i + bs)
│   │   ├── try
│   │   │   └── calls _matmul_chunk(large=large[i:j], small=small, out=out[i:j], direct=direct)
│   │   ├── except torch.cuda.OutOfMemoryError
│   │   │   ├── if num_divide is not None or divides >= max_divide or bs <= 1
│   │   │   │   └── raise
│   │   │   ├── impls increment divides
│   │   │   ├── impls halve bs
│   │   │   ├── impls release cached CUDA memory
│   │   │   └── continue
│   │   └── impls i = j  # advance only after the chunk succeeds
│   └── return  # out, the [N, M] product (large itself when inplace)
├── def _validate_inputs(large: torch.Tensor, small: torch.Tensor, inplace: bool, max_divide: int, num_divide: Optional[int]) -> None
│   ├── assert isinstance(large, torch.Tensor)
│   ├── assert isinstance(small, torch.Tensor)
│   ├── assert large.ndim == 2
│   ├── assert small.ndim == 2
│   ├── assert small.shape[0] == small.shape[1]
│   ├── assert large.shape[1] == small.shape[0]
│   ├── assert large.device == small.device
│   ├── assert large.dtype == small.dtype
│   ├── assert isinstance(inplace, bool)
│   ├── assert isinstance(max_divide, int) and max_divide >= 0
│   ├── assert num_divide is None or (isinstance(num_divide, int) and num_divide >= 0)
│   └── if inplace
│       └── assert not large.requires_grad and not small.requires_grad
└── def _matmul_chunk(large: torch.Tensor, small: torch.Tensor, out: torch.Tensor, direct: bool) -> None
    ├── # Writes large @ small into out for one row-chunk: direct uses out= with no intermediate (out must not alias large); else a temp-copy assignment that is autograd-safe and the only correct form when out aliases large.
    ├── if direct
    │   └── impls torch.matmul(large, small, out=out)
    └── else
        └── impls out[:] = large @ small
```

`utils/ops/dict_ops.py`

```text
dict_ops.py
├── from copy import deepcopy
├── from typing import Any, Dict
└── def merge_missing_dict_values(input_dict: Dict[str, Any], default_dict: Dict[str, Any]) -> Dict[str, Any]
    ├── # Fills missing nested dictionary values from the defaults without overwriting provided values.
    ├── impls recursively merge default_dict into input_dict, copying each value into a new dict via deepcopy
    └── return  # the merged dict
```
