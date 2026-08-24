# criteria/vision_2d/dense_prediction/dense_regression — code structure

## 1. Inheritance / type trees

```text
class DensePredictionCriterion  # the parent this module specializes
└── class DenseRegressionCriterion  # its complete set of direct subclasses, all three of them leaves
    ├── class DepthEstimationCriterion
    ├── class InstanceSegmentationCriterion
    └── class NormalEstimationCriterion
```

## 2. Code structure trees

`criteria/vision_2d/dense_prediction/dense_regression/__init__.py`

```text
__init__.py
└── # Intentionally blank.
```

`criteria/vision_2d/dense_prediction/dense_regression/base.py`

```text
base.py
├── import torch
├── from criteria.vision_2d.dense_prediction.base import DensePredictionCriterion
└── class DenseRegressionCriterion(DensePredictionCriterion)  # its docstring is wrong, documenting a normalize_inputs attribute and __init__ arg on top of the two the signature takes
    ├── # Dense-prediction base for continuous targets: it defaults the ignored value to inf and leaves the per-sample residual to its subclasses.
    ├── def __init__(self, ignore_value: float = float('inf'), reduction: str = 'mean', **kwargs) -> None  [override]
    │   ├── # Defaults the ignored target value to inf, the one value a finite regression target never takes.
    │   └── calls super().__init__(ignore_value=ignore_value, reduction=reduction, **kwargs)
    └── def _compute_unreduced_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor  [override]
        ├── # Each concrete regression criterion owns the residual its own task is scored by.
        └── raise NotImplementedError
```

`criteria/vision_2d/dense_prediction/dense_regression/depth_estimation.py`

```text
depth_estimation.py
├── import torch
├── from criteria.vision_2d.dense_prediction.dense_regression.base import DenseRegressionCriterion
├── from utils.input_checks import check_depth_estimation
└── class DepthEstimationCriterion(DenseRegressionCriterion)
    ├── # Scores an (N, 1, H, W) predicted depth map against its (N, H, W) target, as a masked L1 over the positive-depth pixels.
    ├── def __init__(self, reduction: str = 'mean', **kwargs) -> None  [override]
    │   ├── # Fixes zero as the ignored depth, since a zero depth is how this task encodes an invalid pixel.
    │   └── calls super().__init__(ignore_value=0, reduction=reduction, **kwargs)
    ├── def _task_specific_checks(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None  [override]
    │   ├── # Adds the one depth value check the shared contract leaves open: no predicted depth may be negative.
    │   ├── calls check_depth_estimation(y_pred=y_pred, y_true=y_true)  # the shared depth contract: shapes, dtypes, and an already non-negative target
    │   ├── if any predicted depth is negative
    │   │   └── raise ValueError  # "Predicted depth values must be non-negative"
    │   └── if any ground-truth depth is negative
    │       └── raise ValueError  # "Ground truth depth values must be non-negative" — dead, the shared check asserts this first
    ├── def _get_valid_mask(self, y_true: torch.Tensor) -> torch.Tensor  [override]
    │   ├── # Treats only strictly-positive target depths as supervised, since zero is what encodes an invalid depth.
    │   ├── impls mark the strictly-positive ground-truth depths as the valid pixels
    │   ├── if no pixel is valid
    │   │   └── raise AssertionError  # "All depths in target are invalid"
    │   └── return  # the per-pixel valid mask over the batch
    └── def _compute_unreduced_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor  [override]
        ├── # Averages the absolute depth error over each sample's valid pixels alone.
        ├── impls drop the prediction's singleton channel dimension
        ├── impls take the absolute difference between prediction and ground truth  # impls-node-one-step:skip
        ├── impls zero that difference outside the valid mask
        ├── impls count the valid pixels of each sample
        └── return  # each sample's difference summed over its pixels and divided by that count floored at one
```

`criteria/vision_2d/dense_prediction/dense_regression/instance_segmentation.py`

```text
instance_segmentation.py
├── import torch
├── from criteria.vision_2d.dense_prediction.dense_regression.base import DenseRegressionCriterion
├── from utils.input_checks import check_instance_segmentation
└── class InstanceSegmentationCriterion(DenseRegressionCriterion)
    ├── # Scores an (N, H, W) predicted instance-id map against its (N, H, W) target, as a masked L1 over the non-ignored pixels.
    ├── def __init__(self, ignore_value: int, reduction: str = 'mean', **kwargs) -> None  [override]
    │   ├── # Takes the ignore id from the caller, since which id stands for background or unlabeled is the dataset's choice.
    │   └── calls super().__init__(ignore_value=ignore_value, reduction=reduction, **kwargs)
    ├── def _task_specific_checks(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None  [override]
    │   ├── # Defers entirely to the shared instance-segmentation contract.
    │   └── calls check_instance_segmentation(y_pred=y_pred, y_true=y_true)
    ├── def _get_valid_mask(self, y_true: torch.Tensor) -> torch.Tensor  [override]
    │   ├── # Treats every target pixel other than the ignore id as supervised, and rejects negative ids among them.
    │   ├── impls mark the ground-truth pixels that differ from the ignore value as the valid ones
    │   ├── if no pixel is valid
    │   │   └── raise AssertionError  # "All pixels in target are ignored"
    │   ├── if any valid ground-truth instance id is negative
    │   │   └── raise ValueError  # "Ground truth instance IDs must be non-negative"
    │   └── return  # the per-pixel valid mask over the batch
    └── def _compute_unreduced_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor  [override]
        ├── # Averages the absolute instance-id error over each sample's non-ignored pixels alone.
        ├── impls take the absolute difference between predicted and ground-truth instance ids  # impls-node-one-step:skip
        ├── impls zero that difference outside the valid mask
        ├── impls count the valid pixels of each sample
        └── return  # each sample's difference summed over its pixels and divided by that count floored at one
```

`criteria/vision_2d/dense_prediction/dense_regression/normal_estimation.py`

```text
normal_estimation.py
├── import torch
├── from criteria.vision_2d.dense_prediction.dense_regression.base import DenseRegressionCriterion
├── from utils.input_checks import check_normal_estimation
└── class NormalEstimationCriterion(DenseRegressionCriterion)
    ├── # Scores an (N, 3, H, W) predicted normal map against its (N, 3, H, W) target, as one minus their mean per-pixel dot product over the non-degenerate pixels.
    ├── def __init__(self, reduction: str = 'mean', **kwargs) -> None  [override]
    │   ├── # Fixes zero as the ignored normal, since a zero vector is how this task encodes an invalid pixel.
    │   └── calls super().__init__(ignore_value=0, reduction=reduction, **kwargs)
    ├── def _task_specific_checks(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None  [override]
    │   ├── # Defers entirely to the shared normal-estimation contract.
    │   └── calls check_normal_estimation(y_pred=y_pred, y_true=y_true)
    ├── def _get_valid_mask(self, y_true: torch.Tensor) -> torch.Tensor  [override]
    │   ├── # Treats only target normals of non-zero length as supervised, since a zero vector is what encodes an invalid normal.
    │   ├── impls take each ground-truth normal's length along the channel dimension
    │   └── return  # the per-pixel mask where that length is non-zero
    └── def _compute_unreduced_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor  [override]
        ├── # Turns each sample's mean per-pixel dot product into a cost over its valid pixels, taking both inputs as already unit-length.
        ├── impls take the per-pixel dot product of predicted and ground-truth normals  # impls-node-one-step:skip
        ├── impls zero that cosine map outside the valid mask
        ├── impls count the valid pixels of each sample
        ├── impls average each sample's cosine map over that count floored at one
        └── return  # one minus each sample's mean cosine similarity
```
