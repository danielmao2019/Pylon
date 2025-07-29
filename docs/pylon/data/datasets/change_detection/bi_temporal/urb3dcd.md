# URB3DCD Dataset

## Overview

The URB3DCD (Urban 3D Change Detection) dataset provides high-resolution 3D point cloud data for change detection in urban environments. The dataset contains pairs of LiDAR scans from the same areas captured at different times, with detailed annotations for various types of changes.

## Dataset Information

- **Type**: 3D LiDAR point clouds
- **Task**: Multi-class 3D change detection
- **Number of Classes**: 7
  - Class 0: Unchanged
  - Class 1: Newly Built
  - Class 2: Deconstructed
  - Class 3: New Vegetation
  - Class 4: Vegetation Growth
  - Class 5: Vegetation Removed
  - Class 6: Mobile Objects
- **Splits**: Train, Validation, Test
- **Dataset Versions**: V1 and V2

## Data Structure

The dataset consists of pairs of 3D point clouds with point-wise change labels.

### Input Format

- `pc_0`: First temporal point cloud (with xyz coordinates and features)
- `pc_1`: Second temporal point cloud (with xyz coordinates and features)
- `kdtree_0`: KDTree for efficient neighbor queries on the first point cloud
- `kdtree_1`: KDTree for efficient neighbor queries on the second point cloud

### Label Format

- `change_map`: Point-wise semantic labels for changes (7 classes)

## Usage in Pylon

```python
from data.datasets.change_detection_datasets.bi_temporal import Urb3DCDDataset

# Create dataset with cylinder sampling
dataset = Urb3DCDDataset(
    data_root="/path/to/URB3DCD",
    split="train",
    version=1,             # Dataset version (1 or 2)
    patched=True,          # Whether to use patch-based sampling
    sample_per_epoch=128,  # Number of samples per epoch
    radius=20              # Sampling radius in meters
)

# Get a sample
inputs, labels, meta_info = dataset[0]

# inputs contains 'pc_0', 'pc_1', 'kdtree_0', 'kdtree_1'
# labels contains 'change_map'
```

## Data Preparation

1. Download the dataset (requires registration with the dataset authors)
2. Organize the data according to the expected directory structure:

```
<data_root>/
  ├── IEEE_Dataset_V1/       # Version 1
  │   └── 1-Lidar05/
  │       ├── Test/
  │       ├── TrainLarge-1c/
  │       └── Val/
  └── IEEE_Dataset_V2_Lid05_MS/  # Version 2
      └── Lidar05/
          ├── Test/
          ├── TrainLarge-1c/
          └── Val/
```

3. Create a soft link to the dataset directory in your Pylon project:

```bash
ln -s /path/to/URB3DCD <Pylon_path>/data/datasets/soft_links/URB3DCD
```

## Implementation Details

- The implementation supports both whole scene processing and patch-based sampling
- Three sampling strategies are available:
  - Fixed centers: Pre-determined sampling locations
  - Random centers: Randomly sampled points within the scene
  - Grid centers: Regular grid sampling (default for test/val if no samples per epoch specified)
- Cylinder sampling is used to extract local regions from the point clouds
- KDTrees are used for efficient nearest neighbor queries
- Data normalization is applied to handle varying point densities and scales

## References

- [Dataset Website](http://www.grss-ieee.org/community/technical-committees/information-extraction/urban-3d-challenge/)
- [Paper: Urb3DCD: Urban 3D Change Detection Dataset](https://ieeexplore.ieee.org/document/9553307)

## Research Papers Using This Dataset

- [4D Panoptic LiDAR Segmentation](https://arxiv.org/abs/2102.12580)
- [3D Siamese Network for Urban Change Detection with Multitemporal Point Clouds](https://ieeexplore.ieee.org/document/9553307)
