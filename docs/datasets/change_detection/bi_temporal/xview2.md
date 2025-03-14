# xView2 Dataset

## Overview

The xView2 dataset focuses on building damage assessment in satellite imagery before and after natural disasters. It provides pre-disaster and post-disaster satellite images with corresponding building footprints and damage classification labels.

## Dataset Information

- **Type**: Satellite imagery
- **Task**: Disaster damage assessment and change detection
- **Number of Classes**: Multi-class damage assessment
- **Dataset Size**:
  - Train: 2,799 image pairs
  - Test: 933 image pairs
  - Hold: 933 image pairs (holdout set)

## Data Structure

The dataset consists of high-resolution satellite imagery of pre- and post-disaster scenes with building-level annotations.

### Input Format

- `img_1`: Pre-disaster image
- `img_2`: Post-disaster image

### Label Format

- `lbl_1`: Pre-disaster building footprints
- `lbl_2`: Post-disaster building footprints with damage classification

## Usage in Pylon

```python
from data.datasets.change_detection_datasets.bi_temporal import xView2Dataset

# Create dataset
dataset = xView2Dataset(
    data_root="/path/to/xView2",
    split="train"
)

# Get a sample
inputs, labels, meta_info = dataset[0]

# inputs contains 'img_1' and 'img_2'
# labels contains 'lbl_1' and 'lbl_2'
```

## Data Preparation

1. Download the dataset from the [xView2 website](https://xview2.org/download-links)
2. Extract and organize as follows:

```bash
mkdir <data-root>
cd <data-root>
# download and unzip training set
# Download Challenge training set from xView2 website
tar -xvzf train_images_labels_targets.tar
rm train_images_labels_targets.tar
# download and unzip tier3 set
# Download additional Tier3 training data from xView2 website
tar -xvzf tier3.tar
rm tier3.tar
# download and unzip test set
# Download Challenge test set from xView2 website
tar -xvzf test_images_labels_targets.tar
rm test_images_labels_targets.tar
# download and unzip hold set
# Download Challenge holdout set from xView2 website
tar -xvzf hold_images_labels_targets.tar
rm hold_images_labels_targets.tar
# create a soft-link
ln -s <data-root> <project-root>/data/datasets/soft_links
```

## Implementation Details

- The dataset provides pre- and post-disaster satellite imagery for multiple natural disaster events
- The building damage is classified into multiple levels based on severity
- The implementation organizes the data into train, test, and holdout sets as provided by the xView2 challenge

## References

- [xView2 Challenge Website](https://xview2.org)
- [Dataset Download Links](https://xview2.org/download-links)
- [xView2 Challenge: Building Damage Assessment Paper](https://arxiv.org/abs/1911.09296)

## Research Papers Using This Dataset

- [Change is Everywhere: Single-Temporal Supervised Object Change Detection in Remote Sensing Imagery](https://arxiv.org/abs/2108.07002)
- [Building Damage Detection in Satellite Imagery Using Convolutional Neural Networks](https://arxiv.org/abs/1910.06444)
