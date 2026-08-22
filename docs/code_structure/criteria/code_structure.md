# criteria — code structure

## 1. Inheritance / type trees

```text
class torch.nn.Module
└── class BaseCriterion(torch.nn.Module, ABC)  # from here down, its complete set of direct subclasses within the criteria package
    ├── class SingleTaskCriterion
    ├── class MultiTaskCriterion
    └── class CCDMCriterion
```

## 2. Code structure trees

`criteria/__init__.py`

```text
__init__.py
├── # CRITERIA API
├── from criteria import common, diffusion, vision_2d, vision_3d, wrappers
├── from criteria.base_criterion import BaseCriterion
└── __all__  # Tuple[str, ...] — BaseCriterion plus the common, vision_2d, vision_3d, diffusion, and wrappers subpackages
```

`criteria/base_criterion.py`

```text
base_criterion.py
├── import queue
├── import threading
├── from abc import ABC, abstractmethod
├── from typing import Any, List, Optional
├── import torch
└── class BaseCriterion(torch.nn.Module, ABC)
    ├── # The criterion root: an optional async loss buffer, plus the scoring and summarizing contract every criterion realizes.
    ├── def __init__(self, use_buffer: bool = True) -> None
    │   ├── # Builds the criterion and, when buffering is on, the lock, queue, and daemon worker thread its loss trajectory accumulates through.
    │   ├── calls super().__init__
    │   ├── impls store use_buffer, the flag every buffer path below branches on
    │   ├── if use_buffer
    │   │   ├── impls create _buffer_lock, a threading.Lock
    │   │   ├── impls create _buffer_queue, a queue.Queue
    │   │   ├── impls create _buffer_thread, a daemon threading.Thread targeting _buffer_worker
    │   │   ├── impls start _buffer_thread
    │   │   └── calls self._buffer_worker  # reached indirectly, on the daemon thread just started
    │   └── calls self.reset_buffer
    ├── def reset_buffer(self) -> None
    │   ├── # Empties the accumulated loss trajectory, refusing to run while the worker still has queued values.
    │   ├── if use_buffer
    │   │   ├── assert _buffer_queue is empty  # "Buffer queue is not empty when resetting buffer"
    │   │   └── with _buffer_lock
    │   │       └── impls reset buffer to an empty List[Any]
    │   └── else
    │       └── assert no buffer attribute exists
    ├── def _buffer_worker(self) -> None
    │   ├── # Background thread that drains _buffer_queue for the criterion's lifetime, so the caller never pays for the detach.
    │   └── while True
    │       ├── impls take the next value off _buffer_queue
    │       ├── with _buffer_lock
    │       │   ├── impls detach that value onto CPU
    │       │   └── impls append it to buffer
    │       └── impls mark the _buffer_queue task done
    ├── def add_to_buffer(self, value: torch.Tensor, check_validity: bool = True) -> None
    │   ├── # Hands one scalar loss to the worker thread, so the detach onto CPU happens off the caller's path.
    │   ├── if use_buffer
    │   │   ├── assert a buffer attribute exists
    │   │   ├── assert buffer is a list
    │   │   ├── assert value is a torch.Tensor                            # reporting the type it got
    │   │   ├── assert value is a single-element zero-dimensional tensor  # reporting the shape it got
    │   │   ├── if check_validity
    │   │   │   └── assert value is finite and not NaN  # reporting the offending value
    │   │   └── impls put value on _buffer_queue
    │   └── else
    │       └── assert no buffer attribute exists
    ├── def get_buffer(self) -> List[Any]
    │   ├── # Hands back a snapshot of the accumulated losses under the lock, so a caller reading mid-run never sees a half-written list.
    │   ├── if use_buffer
    │   │   └── with _buffer_lock
    │   │       └── return  # a copy of buffer, the accumulated loss list
    │   └── raise RuntimeError  # "Buffer is not enabled"
    ├── @abstractmethod def __call__(self, y_pred: Any, y_true: Any) -> Any  [abstract]
    │   ├── # Scores one prediction against its ground truth — the one contract every criterion realizes.
    │   └── raise NotImplementedError  # "Abstract method BaseCriterion.__call__ not implemented."
    └── @abstractmethod def summarize(self, output_path: Optional[str] = None) -> Any  [abstract]
        ├── # Reduces the accumulated loss trajectory into this criterion's summary, writing it to output_path when one is given.
        └── raise NotImplementedError  # "Abstract method BaseCriterion.summarize not implemented."
```
