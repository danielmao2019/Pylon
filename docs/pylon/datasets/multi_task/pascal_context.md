# PASCAL Context Dataset

## Overview

The PASCAL Context dataset is an extension of the PASCAL VOC 2010 dataset with additional annotations for scene understanding tasks. It provides dense semantic annotations for the whole scene, including objects and background regions, making it suitable for multi-task learning in scene understanding.

## Dataset Information

- **Type**: Natural scene images
- **Tasks**: 
  - Semantic segmentation (59 or 20 classes)
  - Human parts segmentation
  - Surface normal estimation
  - Saliency detection
  - Edge detection
- **Dataset Size**:
  - Train: 4,998 images
  - Validation: 5,105 images
- **Image Size**: Variable (original)

## Data Structure

The dataset consists of RGB images with corresponding semantic segmentation masks, human parts segmentation, surface normal maps, saliency maps, and edge maps.

### Input Format

- `image`: RGB image of natural scene

### Label Format

- `semantic_segmentation`: Pixel-wise semantic class labels
- `human_parts`: Pixel-wise human body part segmentation
- `normal_estimation`: Surface normal map (3 channels: x, y, z components)
- `saliency`: Saliency map
- `edge_detection`: Binary edge map

## Usage in Pylon

```python
from data.datasets.multi_task_datasets import PascalContextDataset

# Create dataset
dataset = PascalContextDataset(
    data_root="/path/to/PascalContext",
    split="train",
    semantic_granularity="coarse"  # 'coarse' (20 classes) or 'fine' (59 classes)
)

# Get a sample
inputs, labels, meta_info = dataset[0]

# inputs contains 'image'
# labels contains task-specific annotations

# Visualization example
dataset.visualize()
```

## Data Preparation

1. Download the PASCAL VOC 2010 dataset from the [official website](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/)
2. Download the PASCAL Context annotations from [this link](https://cs.stanford.edu/~roozbeh/pascal-context/)
3. Download the multi-task annotations from [this link](https://data.vision.ee.ethz.ch/kmaninis/share/MTL/PASCAL_MT.tgz)
4. Extract and organize the files according to the expected directory structure
5. Create a soft link to the dataset directory in your Pylon project:

```bash
ln -s /path/to/PascalContext <Pylon_path>/data/datasets/soft_links/PascalContext
```

## Implementation Details

- **Semantic Segmentation**:
  - Supports two granularity levels:
    - Fine: 59 classes
    - Coarse: 20 classes (PASCAL VOC classes)
  - Void classes are handled with an ignore index
- **Human Parts Segmentation**:
  - Segments human body into parts (e.g., head, torso, arms, legs)
- **Surface Normal Estimation**:
  - Each pixel has 3 channels representing the x, y, z components of the surface normal
- **Saliency Detection**:
  - Identifies visually salient regions in the image
- **Edge Detection**:
  - Detects object boundaries and contours
- **Visualization**:
  - Includes methods to visualize RGB images and all task-specific annotations

## Research Papers Using This Dataset

- [Multi-Task Attention Network for Lane Detection and Semantic Segmentation](https://ieeexplore.ieee.org/document/9304789)
- [Attentive Single-Tasking of Multiple Tasks](https://arxiv.org/abs/1904.08918)
- [MTI-Net: Multi-Scale Task Interaction Networks for Multi-Task Learning](https://arxiv.org/abs/2001.06902)
- [PAD-Net: Multi-Tasks Guided Prediction-and-Distillation Network for Simultaneous Depth Estimation and Scene Parsing](https://arxiv.org/abs/1805.04409)
- [Cross-Task Attention Mechanism for Dense Multi-Task Learning](https://arxiv.org/abs/2006.01312)
