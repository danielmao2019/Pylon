# NYU-v2 Dataset

## Overview

The NYU-v2 dataset is a widely used indoor scene understanding dataset that provides RGB-D images with multiple annotations for various computer vision tasks. It's particularly popular for multi-task learning research, as it enables joint learning of semantic segmentation, depth estimation, surface normal estimation, and edge detection.

## Dataset Information

- **Type**: Indoor RGB-D scenes
- **Tasks**: 
  - Semantic segmentation (40 or 13 classes)
  - Depth estimation
  - Surface normal estimation
  - Edge detection
- **Dataset Size**:
  - Train: 795 images
  - Validation: 654 images
- **Image Size**: 640×480 pixels (original)

## Data Structure

The dataset consists of RGB images with corresponding semantic segmentation masks, depth maps, surface normal maps, and edge maps.

### Input Format

- `image`: RGB image of indoor scene

### Label Format

- `semantic_segmentation`: Pixel-wise semantic class labels
- `depth_estimation`: Depth map
- `normal_estimation`: Surface normal map (3 channels: x, y, z components)
- `edge_detection`: Binary edge map

## Usage in Pylon

```python
from data.datasets.multi_task_datasets import NYUv2Dataset

# Create dataset
dataset = NYUv2Dataset(
    data_root="/path/to/NYUv2",
    split="train",
    semantic_granularity="coarse"  # 'coarse' (13+1 classes) or 'fine' (40+1 classes)
)

# Get a sample
inputs, labels, meta_info = dataset[0]

# inputs contains 'image'
# labels contains 'semantic_segmentation', 'depth_estimation', 'normal_estimation', 'edge_detection'

# Visualization example
dataset.visualize()
```

## Data Preparation

1. Download the dataset from [this link](https://data.vision.ee.ethz.ch/kmaninis/share/MTL/NYUD_MT.tgz)
2. Extract and organize the files according to the expected directory structure
3. Create a soft link to the dataset directory in your Pylon project:

```bash
ln -s /path/to/NYUv2 <Pylon_path>/data/datasets/soft_links/NYUv2
```

## Implementation Details

- **Semantic Segmentation**:
  - Supports two granularity levels:
    - Fine: 40+1 classes
    - Coarse: 13+1 classes
  - Void classes are handled with an ignore index (250)
- **Depth Estimation**:
  - Depth maps are loaded from .mat files
- **Surface Normal Estimation**:
  - Normal maps are loaded from .mat files
  - Each pixel has 3 channels representing the x, y, z components of the surface normal
- **Edge Detection**:
  - Binary edge maps are loaded from .mat files
- **Visualization**:
  - Includes methods to visualize RGB images, semantic segmentation, depth maps, normal maps, and edge maps

## Research Papers Using This Dataset

- [GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks](https://arxiv.org/pdf/1711.02257.pdf)
- [Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning](https://arxiv.org/pdf/2111.10603.pdf)
- [Conflict-Averse Gradient Descent for Multi-task Learning](https://arxiv.org/pdf/2110.14048.pdf)
- [FAMO: Fast Adaptive Multitask Optimization](https://arxiv.org/pdf/2306.03792.pdf)
- [Towards Impartial Multi-task Learning](https://openreview.net/pdf?id=IMPnRXEWpvr)
- [Multi-Task Learning as a Bargaining Game](https://arxiv.org/pdf/2202.01017.pdf)
- [Independent Component Alignment for Multi-Task Learning](https://arxiv.org/pdf/2305.19000.pdf)
- [Regularizing Deep Multi-Task Networks using Orthogonal Gradients](https://arxiv.org/pdf/1912.06844.pdf)
- [Gradient Surgery for Multi-Task Learning](https://arxiv.org/pdf/2001.06782.pdf)
- [Achievement-based Training Progress Balancing for Multi-Task Learning](https://openaccess.thecvf.com/content/ICCV2023/papers/Yun_Achievement-Based_Training_Progress_Balancing_for_Multi-Task_Learning_ICCV_2023_paper.pdf)
