# CityScapes Dataset

## Overview

The CityScapes dataset is a large-scale dataset containing high-resolution images of urban street scenes. It provides pixel-level annotations for semantic segmentation, instance segmentation, and depth estimation, making it ideal for multi-task learning in autonomous driving and urban scene understanding.

## Dataset Information

- **Type**: Urban street scenes
- **Tasks**: 
  - Semantic segmentation (19 or 7 classes)
  - Instance segmentation
  - Depth estimation
- **Dataset Size**:
  - Train: 2,966 images
  - Validation: 493 images
- **Image Size**: 2048×1024 pixels (original)

## Data Structure

The dataset consists of RGB images with corresponding semantic segmentation, instance segmentation, and depth maps.

### Input Format

- `image`: RGB image of urban street scene

### Label Format

- `semantic_segmentation`: Pixel-wise semantic class labels
- `instance_segmentation`: Pixel-wise instance IDs
- `depth_estimation`: Depth map (disparity)

## Usage in Pylon

```python
from data.datasets.multi_task_datasets import CityScapesDataset

# Create dataset
dataset = CityScapesDataset(
    data_root="/path/to/CityScapes",
    split="train",
    semantic_granularity="coarse"  # 'coarse' (7 classes) or 'fine' (19 classes)
)

# Get a sample
inputs, labels, meta_info = dataset[0]

# inputs contains 'image'
# labels contains 'semantic_segmentation', 'instance_segmentation', 'depth_estimation'

# Visualization example
dataset.visualize()
```

## Data Preparation

1. Download the dataset components from the [official website](https://www.cityscapes-dataset.com/):
   - [Images](https://www.cityscapes-dataset.com/file-handling/?packageID=3)
   - [Segmentation labels](https://www.cityscapes-dataset.com/file-handling/?packageID=1)
   - [Disparity labels](https://www.cityscapes-dataset.com/file-handling/?packageID=7)

2. Extract and organize the files according to the expected directory structure
3. Create a soft link to the dataset directory in your Pylon project:

```bash
ln -s /path/to/CityScapes <Pylon_path>/data/datasets/soft_links/CityScapes
```

## Implementation Details

- **Semantic Segmentation**:
  - Supports two granularity levels:
    - Fine: 19 classes
    - Coarse: 7 classes (flat, construction, nature, vehicle, sky, object, human)
  - Void classes are handled with an ignore index (250)
- **Depth Estimation**:
  - Disparity maps are normalized with mean=0 and std=2729.07
- **Data Filtering**:
  - Some problematic images are automatically filtered out (9 from train, 7 from validation)
- **Visualization**:
  - Includes methods to visualize RGB images, semantic segmentation, and depth maps

## Research Papers Using This Dataset

- [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/pdf/1705.07115.pdf)
- [Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning](https://arxiv.org/pdf/2111.10603.pdf)
- [Conflict-Averse Gradient Descent for Multi-task Learning](https://arxiv.org/pdf/2110.14048.pdf)
- [FAMO: Fast Adaptive Multitask Optimization](https://arxiv.org/pdf/2306.03792.pdf)
- [Towards Impartial Multi-task Learning](https://openreview.net/pdf?id=IMPnRXEWpvr)
- [Multi-Task Learning as a Bargaining Game](https://arxiv.org/pdf/2202.01017.pdf)
- [Multi-Task Learning as Multi-Objective Optimization](https://arxiv.org/pdf/1810.04650.pdf)
- [Independent Component Alignment for Multi-Task Learning](https://arxiv.org/pdf/2305.19000.pdf)
- [Gradient Surgery for Multi-Task Learning](https://arxiv.org/pdf/2001.06782.pdf)
