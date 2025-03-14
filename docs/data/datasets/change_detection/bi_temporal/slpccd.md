# SLPCCD Dataset

## Overview

The Street-Level Point Cloud Change Detection (SLPCCD) dataset is designed for 3D change detection in street-level point clouds. It consists of pairs of point clouds from different time periods (2016 and 2020) with annotated changes.

## Dataset Structure

The SLPCCD dataset is organized as follows:

```
SLPCCD/
├── train.txt
├── val.txt
├── test.txt
└── <scene_directories>/
    ├── point2016.txt
    ├── point2020.txt
    └── point2020_seg.txt (optional)
```

Where:
- `train.txt`, `val.txt`, and `test.txt` contain pairs of file paths to point clouds for each split
- Each line in these files has the format: `<path_to_point2016.txt> <path_to_point2020.txt>`
- Each point cloud file contains XYZ coordinates and optional RGB values
- The `point2020_seg.txt` files contain segmentation/change labels (when available)

## Data Format

### Point Cloud Files
Each point cloud file is a text file with one point per line. The format is:
- Columns 1-3: XYZ coordinates
- Columns 4-6: RGB values (when available)

### Segmentation Files
The segmentation files follow the same format as point cloud files, but include an additional column:
- Column 4: Change label (0 for unchanged, 1 for changed)

## Usage in Pylon

### Loading the Dataset

```python
from data.datasets import SLPCCDDataset

# Create dataset
dataset = SLPCCDDataset(
    data_root="/path/to/SLPCCD",
    split="train",
    num_points=8192,
    use_hierarchy=True
)

# Get a sample
inputs, labels, meta_info = dataset[0]
```

### Configuration Files

Configuration files for training, validation, and testing are available at:
- `configs/common/datasets/change_detection/train/slpccd.py`
- `configs/common/datasets/change_detection/val/slpccd.py`
- `configs/common/datasets/change_detection/test/slpccd.py`

### Dataset Viewer

To visualize the dataset, run the dataset viewer app:

```bash
# Navigate to the Pylon directory
cd /path/to/Pylon

# Run the viewer app
python -m data.datasets.viewer
```

Edit the `dataset_name` variable in `data/datasets/viewer.py` to switch to SLPCCD:

```python
dataset_name = "slpccd"
```

## Data Preparation

If you have the original 3DCDNet dataset, you need to:

1. Download the SLPCCD dataset from [Google Drive](https://drive.google.com/drive/folders/1uf5EmWN2-A5Flcy1l9CUodjEQ2UfUy7u) or [Baiduyun](https://pan.baidu.com/s/1k9DEFLlihKMOEIkUQJ-7TA) (password: 8epz)

2. Create the split files in the following format:
   ```
   # train.txt example
   scene1/point2016.txt scene1/point2020.txt
   scene2/point2016.txt scene2/point2020.txt
   ```

3. Place the dataset files in the directory structure shown above

## Implementation Notes

- The SLPCCD dataset supports hierarchical multi-resolution processing, which is useful for capturing features at different scales.
- The implementation checks for RGB values in the point clouds and uses them as features when available.
- Change labels are loaded from segmentation files when available.

## References

For more information, refer to the original 3DCDNet paper and code:
- Paper: [3DCDNet: Single-shot 3D Change Detection with Point Set Difference Modeling and Dual-path Feature Learning](https://ieeexplore.ieee.org/document/9879908)
- Code: [https://github.com/PointCloudYC/3DCDNet](https://github.com/PointCloudYC/3DCDNet) 