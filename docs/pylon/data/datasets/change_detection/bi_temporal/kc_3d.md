# KC3D Dataset

## Overview

The KC3D dataset is a 3D scene change detection dataset that provides RGB-D image pairs with camera parameters for 3D change localization. It supports research in multi-view 3D change detection and localization.

## Dataset Information

- **Type**: RGB-D imagery with 3D information
- **Task**: 3D change detection and localization
- **Number of Classes**: 2 (changed objects identified by bounding boxes)
- **Dataset Split Options**: Train, Validation, Test

## Data Structure

The dataset consists of paired RGB-D images with camera intrinsics and extrinsics, allowing for 3D registration between views.

### Input Format

- `img_1`, `img_2`: RGB images from two timepoints
- `depth_1`, `depth_2`: Depth maps corresponding to the RGB images
- `intrinsics1`, `intrinsics2`: Camera intrinsic parameters
- `position1`, `position2`: Camera positions
- `rotation1`, `rotation2`: Camera rotation matrices
- `registration_strategy`: Strategy used for registering the two views

### Label Format

- `bbox_1`, `bbox_2`: Bounding boxes of changed objects in each view

## Usage in Pylon

```python
from data.datasets.change_detection_datasets.bi_temporal import KC3DDataset

# Create dataset
dataset = KC3DDataset(
    data_root="/path/to/KC3D",
    split="train",
    use_ground_truth_registration=True
)

# Get a sample
inputs, labels, meta_info = dataset[0]

# inputs contains RGB-D data and camera parameters
# labels contains bounding boxes for changed objects
```

## Data Preparation

1. Download and extract the dataset using the following commands:

```bash
# download
mkdir <data-root>
cd <data-root>
wget https://thor.robots.ox.ac.uk/cyws-3d/kc3d.tar
# extract
tar xvf kc3d.tar
```

2. Create a soft link to the dataset directory in your Pylon project:

```bash
ln -s /path/to/KC3D <Pylon_path>/data/datasets/soft_links/KC3D
```

## Implementation Details

- The dataset supports both ground-truth registration and estimated registration between temporal views
- The implementation provides a utility function to convert segmentation masks to bounding boxes
- The dataset loads RGB images, depth maps, and camera parameters required for 3D change localization
- Each sample provides all necessary information to perform 3D reconstruction and change detection

## References

- [Original Implementation](https://github.com/ragavsachdeva/CYWS-3D/blob/master/kc3d.py)
- [Download Link](https://thor.robots.ox.ac.uk/cyws-3d/kc3d.tar)

## Research Papers Using This Dataset

- [The Change You Want to See (Now in 3D)](https://arxiv.org/abs/2203.09564)
