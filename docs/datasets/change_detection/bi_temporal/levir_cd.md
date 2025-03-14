# LEVIR-CD Dataset

## Overview

The LEVIR Change Detection (LEVIR-CD) dataset is a large-scale remote sensing dataset specifically designed for building change detection. It contains bi-temporal high-resolution (0.5m) Google Earth images with pixel-level annotations of changed buildings.

## Dataset Information

- **Type**: Remote sensing/satellite imagery
- **Task**: Binary change detection (specifically building changes)
- **Number of Classes**: 2 (unchanged, changed)
- **Dataset Size**:
  - Train: 445 image pairs
  - Validation: 64 image pairs
  - Test: 128 image pairs
- **Class Distribution**:
  - Train: Unchanged (445,204,032 pixels), Changed (21,412,334 pixels)
  - Validation: Unchanged (64,292,600 pixels), Changed (2,816,258 pixels)
  - Test: Unchanged (127,380,432 pixels), Changed (6,837,335 pixels)

## Data Structure

The dataset consists of pairs of RGB satellite images with corresponding ground truth masks for building changes.

### Input Format

- `img_1`: First temporal image
- `img_2`: Second temporal image

### Label Format

- `change_map`: Binary mask indicating changed pixels (1) and unchanged pixels (0)

## Usage in Pylon

```python
from data.datasets.change_detection_datasets.bi_temporal import LevirCdDataset

# Create dataset
dataset = LevirCdDataset(
    data_root="/path/to/LEVIR-CD",
    split="train"
)

# Get a sample
inputs, labels, meta_info = dataset[0]

# inputs contains 'img_1' and 'img_2'
# labels contains 'change_map'
```

## Data Preparation

1. Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1dLuzldMRmbBNKPpUkX8Z53hi6NHLrWim)
2. Extract and organize as follows:

```bash
mkdir <data_root_path>
cd <data_root_path>
# unzip the package
unzip val.zip
unzip test.zip
unzip train.zip
rm val.zip
rm test.zip
rm train.zip
# create softlink
ln -s <data_root_path> <Pylon_path>/data/datasets/soft_links/LEVIR-CD
# verify softlink status
stat <Pylon_path>/data/datasets/soft_links/LEVIR-CD
```

## Implementation Details

- The dataset is organized into three splits: train, validation, and test
- Images are loaded as RGB tensors with 3 channels
- The dataset includes a unique SHA1 checksum (`610f742580165b4af94ffae295dbab8986a92b69`) for data verification

## References

- [Original Dataset Source](https://justchenhao.github.io/LEVIR/)
- [Paper: A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection](https://www.mdpi.com/2072-4292/12/10/1662)
- Implementation References:
  - [ChangeStar](https://github.com/Z-Zheng/ChangeStar/blob/master/data/levir_cd/dataset.py)
  - [FTN](https://github.com/AI-Zhpp/FTN/blob/main/data/dataset_swin_levir.py)
  - [MTP](https://github.com/ViTAE-Transformer/MTP/blob/main/RS_Tasks_Finetune/Change_Detection/opencd/datasets/levir_cd.py)
  - [Open-CD](https://github.com/likyoo/open-cd/blob/main/opencd/datasets/levir_cd.py)
  - [CDLab](https://github.com/Bobholamovic/CDLab/blob/master/src/data/levircd.py)
