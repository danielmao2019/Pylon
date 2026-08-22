# criteria/vision_2d/dense_prediction/dense_regression — folder structure

## Code folder structure

```text
criteria/vision_2d/dense_prediction/dense_regression/
├── __init__.py  # intentionally blank
├── base.py      # DenseRegressionCriterion: the dense-regression specialization of the per-pixel loss template, defaulting the ignore value to inf and leaving the unreduced loss to its subclasses
├── depth_estimation.py       # DepthEstimationCriterion: masked L1 over the positive-depth pixels
├── instance_segmentation.py  # InstanceSegmentationCriterion: masked L1 over the non-ignored instance-id pixels
└── normal_estimation.py      # NormalEstimationCriterion: one minus the mean per-pixel dot product over the non-degenerate normal pixels
```

## Tests folder structure

```text
tests/criteria/vision_2d/dense_prediction/dense_regression/
├── test_depth_estimation_criterion.py       # DepthEstimationCriterion: the L1 value, the invalid-depth masking, and its input validation.
├── test_instance_segmentation_criterion.py  # InstanceSegmentationCriterion: the ignore_value setting, the L1 value, the ignored-region masking, and its input validation.
└── test_normal_estimation_criterion.py      # NormalEstimationCriterion: the cosine-similarity loss at its extremes, the degenerate-normal masking, and its input validation.
```
