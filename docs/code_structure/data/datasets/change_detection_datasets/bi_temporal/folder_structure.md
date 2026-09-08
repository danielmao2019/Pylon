# Bi-Temporal Change Detection Datasets Folder Structure

## Code folder structure

```text
data/datasets/change_detection_datasets/bi_temporal/
├── __init__.py
├── air_change_dataset.py
├── cdd_dataset.py
├── kc_3d_dataset.py
├── levir_cd_dataset.py
├── oscd_dataset.py
├── slpccd_dataset.py  # SLPCCD: paired street-level .txt clouds, subsampled to a fixed size and given a KNN hierarchy
├── sysu_cd_dataset.py
├── urb3dcd_dataset.py  # URB3DCD: paired multi-element .ply clouds, patched by cylinder sampling around chosen centers
└── xview2_dataset.py
```

## Tests folder structure

```text
tests/data/datasets/change_detection_datasets/bi_temporal/
├── air_change_dataset/
├── cdd_dataset/
├── kc_3d_dataset/
├── levir_cd_dataset/
├── oscd_dataset/
├── slpccd_dataset/
│   ├── conftest.py
│   ├── test_slpccd_dataset.py  # the datapoint structure every split produces
│   ├── test_slpccd_dataset_version_dict.py
│   └── test_slpccd_dataset_version_hash_discrimination.py
├── sysu_cd_dataset/
├── urb3dcd_dataset/
│   ├── conftest.py
│   ├── test_urb3dcd_dataset.py  # the datapoint structure, the three sampling modes, and the consistency fixed sampling promises
│   ├── test_urb3dcd_dataset_version_dict.py
│   └── test_urb3dcd_dataset_version_hash_discrimination.py
└── xview2_dataset/
```
