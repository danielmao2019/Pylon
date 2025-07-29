# PPSL Dataset

## Overview

The PPSL (Patch-based Pseudo-Siamese Learning) dataset is a synthetic dataset for change detection that creates image pairs by applying transformations and patch-based modifications to source images. It's designed to train models that can detect changes from a single temporal image.

## Dataset Information

- **Type**: Synthetic dataset generated from source images
- **Task**: Binary change detection
- **Methodology**: Applies transformations and patch-based modifications to create synthetic image pairs

## Data Structure

The dataset generates pairs of images with controlled changes and corresponding change maps.

### Input Format

- `img_1`: First image (original with color jitter)
- `img_2`: Second image (from another source with affine transformation and partial patch replacement)

### Label Format

- `change_map`: Binary mask indicating changed pixels (1) and unchanged pixels (0)

## Usage in Pylon

```python
from data.datasets.change_detection_datasets.single_temporal import PPSLDataset

# Create dataset from a source dataset (e.g., a semantic segmentation dataset)
dataset = PPSLDataset(
    data_root="/path/to/source_dataset",
    split="train"
)

# Get a sample
inputs, labels, meta_info = dataset[0]

# inputs contains 'img_1' and 'img_2'
# labels contains 'change_map'

# Visualization example
dataset.visualize(output_dir="./visualization")
```

## Implementation Details

- **Image Transformations**:
  - Color jitter: Applied to the first image (brightness=0.7, contrast=0.7, saturation=0.7, hue=0.2)
  - Affine transformation: Applied to the second image (degrees=±5, scale=1-1.02, translate=±0.02, shear=±5)
- **Patch Replacement**:
  - Replaces the right half of the second image with the right half of the first image
  - Similarly replaces the corresponding label regions
- **Change Map Generation**:
  - Computes the logical XOR between the original label and the patched label
  - Identifies pixels where the semantic class has changed

## Reference Implementation

The implementation is based on the PPSL-MGFDNet repository:

- [Original PPSL Implementation](https://github.com/SGao1997/PPSL_MGFDNet/blob/main/dataset_half_bz24.py)

## Research Papers

- [Pseudo-Siamese Learning for Scene Change Detection Using Unpaired Data](https://ieeexplore.ieee.org/document/9710334)
- [MGFDNet: Multi-Granularity Feature Difference Network for Scene Change Detection](https://ieeexplore.ieee.org/document/9710334)
