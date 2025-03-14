# SYSU-CD Dataset

## Overview

The SYSU-CD dataset is a large-scale change detection dataset derived from high-resolution remote sensing images. It contains multi-temporal image pairs with labeled change maps focusing on land cover and land use changes.

## Dataset Information

- **Type**: Remote sensing satellite imagery
- **Task**: Binary change detection
- **Number of Classes**: 2 (unchanged, changed)
- **Dataset Size**:
  - Train: 12,000 image pairs
  - Validation: 4,000 image pairs
  - Test: 4,000 image pairs
- **Class Distribution**:
  - Train: Unchanged (618,599,552 pixels), Changed (167,833,360 pixels)
  - Validation: Unchanged (205,706,240 pixels), Changed (56,437,744 pixels)
  - Test: Unchanged (200,322,672 pixels), Changed (61,820,912 pixels)

## Data Structure

The dataset consists of RGB imagery with corresponding binary change masks.

### Input Format

- `img_1`: First temporal image (time1 directory)
- `img_2`: Second temporal image (time2 directory)

### Label Format

- `change_map`: Binary mask indicating changed pixels (1) and unchanged pixels (0) (label directory)

## Usage in Pylon

```python
from data.datasets.change_detection_datasets.bi_temporal import SYSU_CD_Dataset

# Create dataset
dataset = SYSU_CD_Dataset(
    data_root="/path/to/SYSU-CD",
    split="train"
)

# Get a sample
inputs, labels, meta_info = dataset[0]

# inputs contains 'img_1' and 'img_2'
# labels contains 'change_map'

# Visualization example
dataset.visualize(output_dir="./visualization")
```

## Data Preparation

1. Download the dataset from [OneDrive](https://mail2sysueducn-my.sharepoint.com/:f:/g/personal/liumx23_mail2_sysu_edu_cn/Emgc0jtEcshAnRkgq1ZTE9AB-kfXzSEzU_PAQ-5YF8Neaw?e=IhVeeZ)
2. Create a soft link to the dataset directory in your Pylon project:

```bash
ln -s /path/to/SYSU-CD <Pylon_path>/data/datasets/soft_links/SYSU-CD
```

## Implementation Details

- The dataset is organized in a directory structure with `time1`, `time2`, and `label` subdirectories for each split
- All images are in PNG format
- The implementation includes a visualization method to display image pairs and change maps
- The dataset has a unique SHA1 checksum (`5e0fa34b0fec61665b62b622da24f17020ec0664`) for data verification

## References

- [Original Dataset Repository](https://github.com/liumency/SYSU-CD)
- [Dataset Download Link](https://mail2sysueducn-my.sharepoint.com/:f:/g/personal/liumx23_mail2_sysu_edu_cn/Emgc0jtEcshAnRkgq1ZTE9AB-kfXzSEzU_PAQ-5YF8Neaw?e=IhVeeZ)

## Research Papers Using This Dataset

- [Deeply Supervised Change Detection for High Resolution Remote Sensing Images](https://ieeexplore.ieee.org/document/8682459)
- [Building Change Detection for Remote Sensing Images Using a Dual-Task Constrained Deep Siamese Convolutional Network Model](https://ieeexplore.ieee.org/document/8821766)
