# OSCD (Onera Satellite Change Detection) Dataset

## Overview

The Onera Satellite Change Detection dataset is a satellite imagery dataset specifically designed for change detection tasks. It consists of pairs of multispectral images taken at different times, with pixel-level change annotations.

## Dataset Information

- **Type**: Remote sensing satellite imagery
- **Task**: Binary change detection
- **Number of Classes**: 2 (unchanged, changed)
- **Dataset Size**:
  - Train: 14 image pairs
  - Test: 10 image pairs
- **Class Distribution**:
  - Train: Unchanged (6,368,388 pixels), Changed (149,190 pixels)
  - Test: Unchanged (2,918,859 pixels), Changed (159,077 pixels)

## Data Structure

The dataset consists of multispectral Sentinel-2 satellite images with 13 spectral bands, though typically only a subset of these bands are used (RGB or a selection of specific bands).

### Input Format

- `img_1`: First temporal image
- `img_2`: Second temporal image

### Label Format

- `change_map`: Binary mask indicating changed pixels (1) and unchanged pixels (0)

## Usage in Pylon

```python
from data.datasets.change_detection_datasets.bi_temporal import OSCDDataset

# Create dataset
dataset = OSCDDataset(
    data_root="/path/to/OSCD",
    split="train",
    bands='3ch'  # Use RGB bands
)

# Get a sample
inputs, labels, meta_info = dataset[0]

# inputs contains 'img_1' and 'img_2'
# labels contains 'change_map'
```

## Data Preparation

1. Download the dataset from the [IEEE Dataport](https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection)
2. Extract and organize as follows:

```bash
mkdir <data-root>
cd <data-root>
# Download the zip files and README.txt from the link above
# Unzip and rename all packages
unzip 'Onera Satellite Change Detection dataset - Images.zip'
rm 'Onera Satellite Change Detection dataset - Images.zip'
mv 'Onera Satellite Change Detection dataset - Images' images
unzip 'Onera Satellite Change Detection dataset - Train Labels.zip'
rm 'Onera Satellite Change Detection dataset - Train Labels.zip'
mv 'Onera Satellite Change Detection dataset - Train Labels' train_labels
unzip 'Onera Satellite Change Detection dataset - Test Labels.zip'
rm 'Onera Satellite Change Detection dataset - Test Labels.zip'
mv 'Onera Satellite Change Detection dataset - Test Labels' test_labels
# Create a soft-link
ln -s <data-root> <project-root>/data/datasets/soft_links
```

## Implementation Details

- The dataset supports different band configurations:
  - `'all'`: All 13 Sentinel-2 bands
  - `'rgb'` or `'3ch'`: Only RGB bands
  - List of specific band indices

## References

- [IEEE Dataport - OSCD](https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection)
- [Original Paper](https://arxiv.org/abs/1810.08468)
- [GitHub Implementations](https://github.com/ServiceNow/seasonal-contrast/blob/main/datasets/oscd_dataset.py)

## Research Papers Using This Dataset

- [Fully Convolutional Siamese Networks for Change Detection](https://ieeexplore.ieee.org/document/8451652)
- [Building Change Detection for Remote Sensing Images Using a Dual-Task Constrained Deep Siamese Convolutional Network Model](https://ieeexplore.ieee.org/document/8821766)
- [FGCN: A Feature-Refined Graph Convolutional Network for Semantic Segmentation of Very High Resolution Remote Sensing Images](https://ieeexplore.ieee.org/document/9667512)
