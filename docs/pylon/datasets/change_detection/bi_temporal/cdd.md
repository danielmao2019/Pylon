# CDD (Change Detection Dataset)

## Overview

The Change Detection Dataset (CDD) is a large-scale dataset designed for the change detection task in remote sensing imagery. It contains pairs of aerial or satellite images taken at different times, with pixel-level change annotations.

## Dataset Information

- **Type**: Remote sensing/aerial imagery
- **Task**: Binary change detection
- **Number of Classes**: 2 (unchanged, changed)
- **Dataset Size**:
  - Train: 26,000 image pairs
  - Validation: 6,998 image pairs
  - Test: 7,000 image pairs
- **Class Distribution**:
  - Train: Unchanged (1,540,513,402 pixels), Changed (163,422,598 pixels)
  - Validation: Unchanged (414,542,888 pixels), Changed (44,078,040 pixels)
  - Test: Unchanged (413,621,824 pixels), Changed (45,130,176 pixels)

## Data Structure

The dataset consists of RGB imagery with significant change and no-change regions.

### Input Format

- `img_1`: First temporal image
- `img_2`: Second temporal image

### Label Format

- `change_map`: Binary mask indicating changed pixels (1) and unchanged pixels (0)

## Usage in Pylon

```python
from data.datasets.change_detection_datasets.bi_temporal import CDDDataset

# Create dataset
dataset = CDDDataset(
    data_root="/path/to/CDD",
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

1. Download the dataset from [Google Drive](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9/edit)
2. Extract and organize as follows:

```bash
mkdir <data-root>
cd <data-root>
# Download the zip files and extract them
unrar x ChangeDetectionDataset.rar
# Create a soft link
ln -s <data_root_path> <Pylon_path>/data/datasets/soft_links/CDD
# Verify the soft link
stat <Pylon_path>/data/datasets/soft_links/CDD
```

## Implementation Details

- The dataset is organized into train, validation, and test splits
- The implementation includes visualization utilities to display image pairs and change maps
- Images are loaded as RGB tensors with 3 channels

## References

- [Original Dataset Source](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9/edit)
- Implementation References:
  - [Seasonal Contrast](https://github.com/ServiceNow/seasonal-contrast/blob/main/datasets/oscd_dataset.py)
  - [Fabric](https://github.com/granularai/fabric/blob/igarss2019/utils/dataloaders.py)
  - [UNet-LSTM](https://github.com/NIX369/UNet_LSTM/blob/master/custom.py)
  - [DINO-MC](https://github.com/WennyXY/DINO-MC/blob/main/data_process/oscd_dataset.py)
