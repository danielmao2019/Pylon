# criteria/vision_2d/dense_prediction — code structure

## 1. Inheritance / type trees

```text
class BaseCriterion  # the ancestor chain down to this module, one traced path
└── class SingleTaskCriterion  # its __call__ is what invokes the _compute_loss hook below
    └── class DensePredictionCriterion  # its two direct subclasses, each of which has concrete subclasses of its own
        ├── class DenseClassificationCriterion
        └── class DenseRegressionCriterion
```

## 2. Code structure trees

`criteria/vision_2d/dense_prediction/__init__.py`

```text
__init__.py
└── # Intentionally blank.
```

`criteria/vision_2d/dense_prediction/base.py`

```text
base.py
├── from typing import Optional, Union
├── from abc import abstractmethod
├── import torch
├── import torchvision.transforms.functional as F
├── from criteria.wrappers import SingleTaskCriterion
└── class DensePredictionCriterion(SingleTaskCriterion)
    ├── # Template for every per-pixel loss: it owns the ignore value, the resolution matching, and the batch reduction, leaving _compute_loss's three task hooks to subclasses.
    ├── REDUCTION_OPTIONS  # List[str] — the two batch reductions __init__ accepts, mean and sum
    ├── def __init__(self, ignore_value: Optional[Union[int, float]] = None, reduction: str = 'mean', **kwargs) -> None  [override]
    │   ├── # Fixes the two settings every dense-prediction loss shares: which target value is ignored, and how per-sample losses reduce over the batch.
    │   ├── calls super().__init__(**kwargs)
    │   ├── if no ignore value was given
    │   │   └── raise ValueError  # "Child classes must provide a default ignore_value if None is passed"
    │   ├── if the ignore value is not a number
    │   │   └── raise ValueError  # "ignore_value must be a number", reporting the type it got
    │   ├── impls store the ignore value
    │   ├── if the reduction is not one of REDUCTION_OPTIONS
    │   │   └── raise ValueError  # "reduction must be one of REDUCTION_OPTIONS", reporting the one it got
    │   └── impls store the reduction
    ├── def _compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor  # -> a scalar loss; SingleTaskCriterion.__call__ reaches it through a hasattr check
    │   ├── # Runs the fixed per-pixel sequence — validate, match resolution, mask, score per sample, reduce — that every subclass plugs its three hooks into.
    │   ├── calls self._task_specific_checks
    │   ├── calls self._match_resolution(y_pred, y_true)                              # -> the ground truth at the prediction's spatial resolution
    │   ├── assert prediction and ground truth agree on batch size                    # "Batch size mismatch", reporting both batch sizes
    │   ├── assert prediction and ground truth agree on spatial dimensions            # "Spatial dimensions mismatch", reporting both spatial shapes
    │   ├── calls self._get_valid_mask(y_true)                                        # -> the per-pixel valid mask
    │   ├── assert the valid mask carries the ground truth's batch and spatial shape  # "Invalid mask shape", reporting the expected and actual shapes
    │   ├── calls self._compute_unreduced_loss(y_pred, y_true, valid_mask)            # -> one loss per sample
    │   ├── assert the unreduced loss holds exactly one entry per sample              # "Unreduced loss should have shape (N,)", reporting the actual shape
    │   ├── if the reduction is 'mean'
    │   │   └── return  # the mean over the per-sample losses
    │   └── else
    │       └── return  # the sum over the per-sample losses
    ├── def _match_resolution(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor
    │   ├── # Resamples the target onto the prediction's spatial size when the two disagree, so every downstream step compares pixel to pixel.
    │   ├── if the ground truth's spatial dimensions differ from the prediction's
    │   │   ├── impls choose nearest interpolation when the ground truth is an int64 tensor and bilinear otherwise  # impls-node-one-step:skip
    │   │   └── calls F.resize(y_true, size=y_pred.shape[-2:], interpolation=getattr(F.InterpolationMode, mode.upper()))
    │   └── return  # the ground truth at the prediction's spatial resolution
    ├── @abstractmethod def _task_specific_checks(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None  [abstract]
    │   ├── # Hook where each task asserts its own prediction and target shape and value contract, before any loss is computed.
    │   └── raise NotImplementedError
    ├── @abstractmethod def _get_valid_mask(self, y_true: torch.Tensor) -> torch.Tensor  [abstract]  # -> an (N, H, W) bool mask
    │   ├── # Hook where each task says which pixels carry supervision, so the loss averages over those alone.
    │   └── raise NotImplementedError
    └── @abstractmethod def _compute_unreduced_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor  [abstract]  # valid_mask (N, H, W) bool -> (N,) per-sample losses
        ├── # Hook where each task scores its own per-sample residual over the valid pixels, ahead of the batch reduction.
        └── raise NotImplementedError
```
