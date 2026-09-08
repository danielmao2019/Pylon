# PCR Datasets Folder Structure

## Code folder structure

```text
data/datasets/pcr_datasets/
├── __init__.py
├── base_pcr_dataset.py                 # BasePCRDataset: the shared input/label names and the whole viewer display for a registration pair
├── kitti_dataset.py                    # KITTI: velodyne scan pairs, their ground truth refined by ICP and cached per pair
├── modelnet40_dataset.py               # ModelNet40: OFF meshes self-registered through the synthetic-transform base, cropped by RandomPointCrop
├── synthetic_transform_pcr_dataset.py  # SyntheticTransformPCRDataset: one cloud posed against a sampled transform, searched by trial until the overlap lands in range, at the width that transform carries
└── threedmatch_dataset.py              # 3DMatch and 3DLoMatch: annotated .ply fragment pairs, split by the overlap band each names, at the width their registration transforms carry
```

## Tests folder structure

```text
tests/data/datasets/pcr_datasets/
├── conftest.py  # the per-dataset config fixtures every suite here builds its dataset from
├── kitti_dataset/
│   ├── test_kitti_dataset.py  # the datapoint structure every KITTI split produces
│   ├── test_kitti_dataset_version_dict.py
│   └── test_kitti_dataset_version_hash_discrimination.py
├── modelnet40_dataset/
├── synthetic_transform_pcr_dataset/
└── threedmatch_dataset/
    └── test_threedmatch_dataset.py  # the datapoint structure both 3DMatch splits produce, and the overlap their annotations claim
```
