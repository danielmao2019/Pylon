# I3PE Dataset

## Overview

The I3PE (Intra-Image Patch Exchange and Inter-Image Patch Exchange) dataset is a synthetic dataset for change detection that creates pairs of images with controlled changes. It uses semantic segmentation and patch exchange techniques to generate realistic change maps between images.

## Dataset Information

- **Type**: Synthetic dataset generated from source images
- **Task**: Binary change detection
- **Methodology**: Uses two techniques to generate changes:
  - Intra-Image Patch Exchange: Swaps patches within the same image
  - Inter-Image Patch Exchange: Swaps patches between different images

## Data Structure

The dataset generates pairs of images by exchanging patches and creates corresponding change maps.

### Input Format

- `img_1`: First image (original)
- `img_2`: Second image (with patches exchanged)

### Label Format

- `change_map`: Binary mask indicating changed pixels (1) and unchanged pixels (0)

## Usage in Pylon

```python
from data.datasets.change_detection_datasets.single_temporal import I3PEDataset

# Create dataset from a source dataset (e.g., ImageNet)
dataset = I3PEDataset(
    exchange_ratio=0.75,  # Ratio of patches to exchange
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

- **Image Segmentation**: Uses SLIC (Simple Linear Iterative Clustering) to segment the source image into 1000 regions
- **Image Clustering**: Uses DBSCAN to cluster similar regions together (with eps=7.0, min_samples=10)
- **Patch Exchange**:
  - Randomly selects a patch size from [16, 32, 64, 128] pixels
  - Exchanges approximately 75% of patches (configurable via exchange_ratio)
  - Creates corresponding change maps showing where patches were exchanged
- **Visualization**: Includes methods to visualize segmentation, clustering, and changes

## Reference Implementation

The implementation is based on the official I3PE repository:

- [Original I3PE Repository](https://github.com/ChenHongruixuan/I3PE)
- [Dataset Implementation](https://github.com/ChenHongruixuan/I3PE/blob/master/data/datasets.py)
- [Clustering Implementation](https://github.com/ChenHongruixuan/I3PE/blob/master/data/generate_clustering_results.py)
- [Object Generation](https://github.com/ChenHongruixuan/I3PE/blob/master/data/generate_object.py)

## Research Papers

- [I3PE: Implicit Change Prompt Learning for Remote Sensing Change Detection](https://www.mdpi.com/2072-4292/15/8/2068)
- [Learning to Detect Changes from a Single Temporal Remote Sensing Image](https://www.mdpi.com/2072-4292/15/1/211)
