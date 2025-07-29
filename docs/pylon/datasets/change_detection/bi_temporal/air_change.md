# AirChange Dataset

## Overview

The SZTAKI AirChange Benchmark dataset is designed for aerial image change detection, consisting of pairs of aerial images with change annotations. The dataset features urban and rural areas with building changes.

## Dataset Information

- **Type**: Aerial imagery
- **Task**: Binary change detection
- **Number of Classes**: 2 (unchanged, changed)
- **Dataset Size**:
  - Train: 3,744 image crops (generated from 12 base images)
  - Test: 12 full images
- **Image Size**:
  - Full images: 952×640 pixels
  - Test crops: 784×448 pixels
  - Train crops: 112×112 pixels
- **Class Distribution**:
  - Train: Unchanged (~44,702,892 pixels), Changed (~2,261,831 pixels)
  - Test: Unchanged (4,016,897 pixels), Changed (197,887 pixels)

## Data Structure

The dataset consists of aerial image pairs with corresponding ground truth masks where changes (primarily building changes) are marked.

### Input Format

- `img_1`: First temporal image
- `img_2`: Second temporal image

### Label Format

- `change_map`: Binary mask indicating changed pixels (1) and unchanged pixels (0)

## Usage in Pylon

```python
from data.datasets.change_detection_datasets.bi_temporal import AirChangeDataset

# Create dataset
dataset = AirChangeDataset(
    data_root="/path/to/AirChange",
    split="train"
)

# Get a sample
inputs, labels, meta_info = dataset[0]

# inputs contains 'img_1' and 'img_2'
# labels contains 'change_map'
```

## Data Preparation

1. Download the dataset using the following commands:

```bash
wget http://mplab.sztaki.hu/~bcsaba/test/SZTAKI_AirChange_Benchmark.zip
unzip SZTAKI_AirChange_Benchmark.zip
mv SZTAKI_AirChange_Benchmark AirChange
rm SZTAKI_AirChange_Benchmark.zip
```

2. Create a soft link to the dataset directory in your Pylon project:

```bash
ln -s /path/to/AirChange <Pylon_path>/data/datasets/soft_links/AirChange
```

## Implementation Details

- The implementation creates training samples by cropping the full images into smaller 112×112 patches
- For training, 312 crops are generated from each base image (3,744 total crops from 12 images)
- A special L-shaped cropping strategy is used to create diverse training samples
- Test evaluation is performed on larger 784×448 crops from the original images

## References

- [SZTAKI AirChange Benchmark Dataset](http://mplab.sztaki.hu/~bcsaba/test/SZTAKI_AirChange_Benchmark.zip)
- [Dataset Description](http://web.eee.sztaki.hu/remotesensing/airchange_benchmark.html)

## Research Papers Using This Dataset

- [Change Detection Based on Deep Siamese Convolutional Network for Optical Aerial Images](https://ieeexplore.ieee.org/document/8451652)
- [Fully Convolutional Siamese Networks for Change Detection](https://ieeexplore.ieee.org/document/8451652)
