# criteria/vision_2d/dense_prediction — folder structure

## Code folder structure

```text
criteria/vision_2d/dense_prediction/
├── __init__.py  # intentionally blank
├── base.py      # DensePredictionCriterion: the per-pixel loss template — ignore-value and reduction settings, ground-truth resolution matching, and the validate → mask → unreduced-loss → reduce sequence
├── dense_classification/  # per-pixel classification criteria
└── dense_regression/      # per-pixel regression criteria
```

## Tests folder structure

```text
tests/criteria/vision_2d/dense_prediction/
├── dense_classification/  # tests of the per-pixel classification criteria
└── dense_regression/      # tests of the per-pixel regression criteria
```
