# criteria/vision_2d/dense_prediction/dense_regression — tests structure

## 1. Tests structure trees

`tests/criteria/vision_2d/dense_prediction/dense_regression/test_depth_estimation_criterion.py`

```text
test_depth_estimation_criterion.py
├── import pytest
├── import torch
├── import numpy as np
├── from criteria.vision_2d.dense_prediction.dense_regression.depth_estimation import DepthEstimationCriterion
├── def test_depth_estimation_basic()
│   ├── # Random positive depths score to a non-negative scalar, the shape every training step consumes.
│   ├── calls DepthEstimationCriterion  # -> the criterion under test
│   ├── impls a (2, 1, 4, 4) batch of random predicted depths scaled to [0, 10)
│   ├── impls a matching batch of random positive ground-truth depths
│   ├── calls criterion.__call__  # prediction and ground truth -> the loss
│   ├── assert the loss is a torch.Tensor
│   ├── assert the loss is zero-dimensional
│   └── assert the loss is non-negative
├── def test_depth_estimation_perfect_predictions()
│   ├── # A prediction equal to the ground truth costs nothing, pinning the L1 residual's zero.
│   ├── calls DepthEstimationCriterion  # -> the criterion under test
│   ├── impls a (2, 4, 4) batch of random ground-truth depths scaled to [0, 10)
│   ├── impls the prediction as that ground truth carrying a channel dimension
│   ├── calls criterion.__call__  # prediction and ground truth -> the loss
│   └── assert the loss is below 1e-6
├── def test_depth_estimation_with_invalid_depths()
│   ├── # Zero-depth pixels are masked out rather than scored, so a target carrying some still yields a finite loss.
│   ├── calls DepthEstimationCriterion  # -> the criterion under test
│   ├── impls a (2, 4, 4) batch of random ground-truth depths scaled to [0, 10)
│   ├── impls zero one of those depths, making it invalid
│   ├── impls a matching batch of random positive predicted depths, one channel
│   ├── calls criterion.__call__  # prediction and ground truth -> the loss
│   ├── assert the loss is a torch.Tensor
│   ├── assert the loss is zero-dimensional
│   └── assert the loss is non-negative
├── def test_depth_estimation_all_invalid()
│   ├── # A target whose depths are all zero leaves nothing to supervise, so the criterion refuses it instead of averaging over an empty mask.
│   ├── calls DepthEstimationCriterion  # -> the criterion under test
│   ├── impls an all-zero batch of ground-truth depths
│   ├── impls a matching batch of random predicted depths, one channel
│   └── with pytest.raises(AssertionError)
│       └── calls criterion.__call__  # prediction and ground truth
└── def test_depth_estimation_input_validation()
    ├── # The depth contract is enforced before any loss is computed: one prediction channel, matching spatial dimensions, and non-negative depths.
    ├── calls DepthEstimationCriterion  # -> the criterion under test
    ├── with pytest.raises(AssertionError)
    │   ├── impls a prediction carrying two channels instead of one
    │   ├── impls a matching ground truth
    │   └── calls criterion.__call__  # prediction and ground truth
    ├── with pytest.raises(AssertionError)
    │   ├── impls a well-shaped prediction
    │   ├── impls a ground truth of a different width
    │   └── calls criterion.__call__  # prediction and ground truth
    └── with pytest.raises(AssertionError)
        ├── impls a positive prediction
        ├── impls a ground truth of negative depths
        └── calls criterion.__call__  # prediction and ground truth
```

`tests/criteria/vision_2d/dense_prediction/dense_regression/test_instance_segmentation_criterion.py`

```text
test_instance_segmentation_criterion.py
├── import pytest
├── import torch
├── import numpy as np
├── from criteria.vision_2d.dense_prediction.dense_regression.instance_segmentation import InstanceSegmentationCriterion
├── def test_instance_segmentation_init()
│   ├── # The caller's own ignore id is the one this criterion masks by, whichever value it is.
│   ├── calls InstanceSegmentationCriterion(ignore_value=-1)
│   ├── assert the criterion kept that ignore id
│   ├── calls InstanceSegmentationCriterion(ignore_value=255)
│   └── assert the criterion kept that ignore id
├── def test_instance_segmentation_basic()
│   ├── # A uniform random prediction against random instance ids scores to a non-negative scalar.
│   ├── calls InstanceSegmentationCriterion  # -> the criterion under test
│   ├── impls a (2, 4, 4) batch of uniform random predictions
│   ├── impls a matching batch of random ground-truth instance ids
│   ├── calls criterion.__call__  # prediction and ground truth -> the loss
│   ├── assert the loss is a torch.Tensor
│   ├── assert the loss is zero-dimensional
│   └── assert the loss is non-negative
├── def test_instance_segmentation_perfect_predictions()
│   ├── # A prediction equal to the ground truth ids costs nothing, pinning the L1 residual's zero.
│   ├── calls InstanceSegmentationCriterion  # -> the criterion under test
│   ├── impls a (2, 4, 4) batch of random ground-truth instance ids in [0, 10)
│   ├── impls the prediction as that ground truth in floating point
│   ├── calls criterion.__call__  # prediction and ground truth -> the loss
│   └── assert the loss is below 1e-6
├── def test_instance_segmentation_with_ignored_regions()
│   ├── # Pixels carrying the ignore id are masked out rather than scored, so a target carrying some still yields a finite loss.
│   ├── calls InstanceSegmentationCriterion  # -> the criterion under test
│   ├── impls a (2, 4, 4) batch of random ground-truth instance ids in [0, 10)
│   ├── impls set one pixel of the first sample to the ignore id
│   ├── impls set one pixel of the second sample to the ignore id
│   ├── impls a matching batch of uniform random predictions
│   ├── calls criterion.__call__  # prediction and ground truth -> the loss
│   ├── assert the loss is a torch.Tensor
│   ├── assert the loss is zero-dimensional
│   └── assert the loss is non-negative
├── def test_instance_segmentation_all_ignored()
│   ├── # A target that is entirely the ignore id leaves nothing to supervise, so the criterion refuses it instead of averaging over an empty mask.
│   ├── calls InstanceSegmentationCriterion  # -> the criterion under test
│   ├── impls a ground truth filled entirely with the ignore id
│   ├── impls a matching batch of uniform random predictions
│   └── with pytest.raises(AssertionError)
│       └── calls criterion.__call__  # prediction and ground truth
└── def test_instance_segmentation_input_validation()
    ├── # The instance-id contract is enforced before any loss is computed: matching spatial dimensions, matching batch size, and three-dimensional predictions.
    ├── calls InstanceSegmentationCriterion  # -> the criterion under test
    ├── with pytest.raises(AssertionError)
    │   ├── impls a well-shaped prediction
    │   ├── impls a ground truth of a different width
    │   └── calls criterion.__call__  # prediction and ground truth
    ├── with pytest.raises(AssertionError)
    │   ├── impls a well-shaped prediction
    │   ├── impls a ground truth of a different batch size
    │   └── calls criterion.__call__  # prediction and ground truth
    └── with pytest.raises(AssertionError)
        ├── impls a prediction carrying a channel dimension it should not have
        ├── impls a well-shaped ground truth
        └── calls criterion.__call__  # prediction and ground truth
```

`tests/criteria/vision_2d/dense_prediction/dense_regression/test_normal_estimation_criterion.py`

```text
test_normal_estimation_criterion.py
├── import pytest
├── import torch
├── from criteria.vision_2d.dense_prediction.dense_regression.normal_estimation import NormalEstimationCriterion
├── def test_normal_estimation_basic()
│   ├── # Random normals score to a zero-dimensional scalar; only the ground truth is rescaled, so the [0, 2] bound is empirical rather than contractual.
│   ├── calls NormalEstimationCriterion  # -> the criterion under test
│   ├── impls a (2, 3, 4, 4) batch of random predicted normals
│   ├── impls a matching batch of random three-channel ground-truth normals
│   ├── impls rescale those ground-truth normals to unit length
│   ├── calls criterion.__call__  # prediction and ground truth -> the loss
│   ├── assert the loss is a torch.Tensor
│   ├── assert the loss is zero-dimensional
│   └── assert the loss lies in [0, 2]
├── def test_normal_estimation_perfect_predictions()
│   ├── # Predicting the ground-truth normals gives cosine similarity one, pinning the loss's zero end.
│   ├── calls NormalEstimationCriterion  # -> the criterion under test
│   ├── impls a (2, 3, 4, 4) batch of random ground-truth normals
│   ├── impls rescale those normals to unit length
│   ├── impls the prediction as a copy of that ground truth
│   ├── calls criterion.__call__  # prediction and ground truth -> the loss
│   └── assert the loss is within 1e-5 of zero
├── def test_normal_estimation_opposite_predictions()
│   ├── # Predicting the negated normals gives cosine similarity minus one, pinning the loss's other end at two.
│   ├── calls NormalEstimationCriterion  # -> the criterion under test
│   ├── impls a (2, 3, 4, 4) batch of random ground-truth normals
│   ├── impls rescale those normals to unit length
│   ├── impls the prediction as the negation of that ground truth
│   ├── calls criterion.__call__  # prediction and ground truth -> the loss
│   └── assert the loss is within 1e-5 of two
├── def test_normal_estimation_with_invalid_normals()
│   ├── # Zero-vector pixels are masked out rather than scored; with only the ground truth rescaled, the [0, 2] bound is again empirical.
│   ├── calls NormalEstimationCriterion  # -> the criterion under test
│   ├── impls a (2, 3, 4, 4) batch of random ground-truth normals
│   ├── impls zero one pixel's normal across the batch, making it degenerate
│   ├── impls take every ground-truth normal's length, the zeroed ones included
│   ├── impls rescale only the non-degenerate normals to unit length
│   ├── impls a matching batch of random three-channel predicted normals
│   ├── calls criterion.__call__  # prediction and ground truth -> the loss
│   ├── assert the loss is a torch.Tensor
│   ├── assert the loss is zero-dimensional
│   └── assert the loss lies in [0, 2]
└── def test_normal_estimation_input_validation()
    ├── # The normal contract is enforced before any loss is computed: three channels, matching spatial dimensions, and four-dimensional inputs.
    ├── calls NormalEstimationCriterion  # -> the criterion under test
    ├── with pytest.raises(AssertionError)
    │   ├── impls a prediction carrying four channels instead of three
    │   ├── impls a well-shaped ground truth
    │   └── calls criterion.__call__  # prediction and ground truth
    ├── with pytest.raises(AssertionError)
    │   ├── impls a well-shaped prediction
    │   ├── impls a ground truth of a different width
    │   └── calls criterion.__call__  # prediction and ground truth
    └── with pytest.raises(AssertionError)
        ├── impls a prediction missing its batch or spatial dimension
        ├── impls a well-shaped ground truth
        └── calls criterion.__call__  # prediction and ground truth
```
