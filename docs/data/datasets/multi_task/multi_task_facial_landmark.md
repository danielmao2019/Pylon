# Multi-Task Facial Landmark Dataset

## Overview

The Multi-Task Facial Landmark dataset is designed for facial landmark detection and related facial attribute prediction tasks. It provides facial images with annotations for landmark positions and multiple binary attributes, enabling multi-task learning approaches to facial analysis.

## Dataset Information

- **Type**: Facial images with landmark and attribute annotations
- **Tasks**: 
  - Facial landmark detection (5 landmarks)
  - Gender classification
  - Smile detection
  - Glasses detection
  - Head pose estimation
- **Dataset Size**:
  - Train and test splits available (sizes vary by version)

## Data Structure

The dataset consists of facial images with landmark coordinates and binary attribute labels.

### Input Format

- `image`: RGB facial image

### Label Format

- `landmarks`: 5 facial landmark coordinates (10 values: x1, y1, x2, y2, x3, y3, x4, y4, x5, y5)
- `gender`: Binary gender classification (0: female, 1: male)
- `smile`: Binary smile detection (0: not smiling, 1: smiling)
- `glasses`: Binary glasses detection (0: no glasses, 1: wearing glasses)
- `pose`: Head pose classification

## Usage in Pylon

```python
from data.datasets.multi_task_datasets import MultiTaskFacialLandmarkDataset

# Create dataset
dataset = MultiTaskFacialLandmarkDataset(
    data_root="/path/to/MTFL",
    split="train"
)

# Get a sample
inputs, labels, meta_info = dataset[0]

# inputs contains 'image'
# labels contains 'landmarks', 'gender', 'smile', 'glasses', 'pose'
```

## Data Preparation

1. Download the dataset from the [official website](https://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html)
2. Extract and organize the files according to the expected directory structure
3. Create a soft link to the dataset directory in your Pylon project:

```bash
ln -s /path/to/MTFL <Pylon_path>/data/datasets/soft_links/MTFL
```

## Implementation Details

- **Landmark Format**:
  - 5 facial landmarks (eye centers, nose tip, mouth corners)
  - Coordinates are stored as (x1, y1, x2, y2, x3, y3, x4, y4, x5, y5)
- **Data Loading**:
  - Images and labels are loaded from text files in the dataset directory
  - Each line in the text file contains the image path followed by landmark coordinates and attribute labels

## Research Papers Using This Dataset

- [Facial Landmark Detection by Deep Multi-task Learning](https://link.springer.com/chapter/10.1007/978-3-319-10599-4_7)
- [Learning Deep Representation for Face Alignment with Auxiliary Attributes](https://ieeexplore.ieee.org/document/7299099)
- [Tasks Consolidated: On the Effective Use of Auxiliary Tasks for Multi-Task Learning](https://arxiv.org/abs/2106.04171)
