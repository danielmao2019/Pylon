# criteria/vision_2d — folder structure

## Code folder structure

```text
criteria/vision_2d/
├── __init__.py        # exposes the dense-prediction base classes, every concrete dense criterion, and the change_detection subpackage
├── change_detection/  # change-detection criteria
└── dense_prediction/  # per-pixel dense prediction criteria
```

## Tests folder structure

```text
tests/criteria/vision_2d/
├── change_detection/  # tests of the change-detection criteria
└── dense_prediction/  # tests of the per-pixel dense prediction criteria
```
