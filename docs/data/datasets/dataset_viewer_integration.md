# Dataset Viewer Integration Guide

## Overview

This guide covers integrating your dataset implementation with Pylon's data viewer for visual inspection and debugging.

## Quick Setup

### 1. Add to Dataset Groups

Update `/data/viewer/backend/backend.py` to include your dataset in the appropriate group:

```python
DATASET_GROUPS = {
    'semseg': ['coco_stuff_164k'],
    '2dcd': ['air_change', 'cdd', 'levir_cd', 'oscd', 'sysu_cd'],
    '3dcd': ['urb3dcd', 'slpccd'],
    'pcr': ['synth_pcr', 'real_pcr', 'kitti', 'your_new_dataset'],  # Add here
}
```

**Dataset Type Guidelines**:
- `semseg`: Semantic segmentation datasets (single image → label map)
- `2dcd`: 2D change detection datasets (two images → change map)  
- `3dcd`: 3D change detection datasets (two point clouds → change map)
- `pcr`: Point cloud registration datasets (two point clouds → transformation)

### 2. Create Viewer Configuration

Create a config file in the appropriate directory:

**Path Pattern**: `/configs/common/datasets/{domain}/train/{dataset_name}_data_cfg.py`

**Example for PCR dataset**:
```python
# /configs/common/datasets/point_cloud_registration/train/your_dataset_data_cfg.py
import torch
import data
import utils

data_cfg = {
    'train_dataset': {
        'class': data.datasets.YourDataset,
        'args': {
            'data_root': './data/datasets/soft_links/your_data',
            'split': 'train',
            # Add dataset-specific parameters
        },
    },
    'train_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 1,
            'num_workers': 4,
            'shuffle': True,
        },
    },
    'criterion': None,  # Not needed for viewer
}
```

**Important Notes**:
- **File name must match pattern**: `{dataset_name}_data_cfg.py`
- **Config structure must match** the training config format

### 3. Test Integration

Launch the viewer and verify your dataset appears:

```bash
# Launch data viewer
python -m data.viewer.cli

# Your dataset should appear in the dropdown as:
# [PCR] your_dataset_name
```

**Troubleshooting**:
- **Dataset not appearing**: Check config file name and location
- **Loading errors**: Verify data paths and dataset parameters
- **Display issues**: Ensure data format matches viewer expectations
- **Import errors**: Check that all required modules are imported in config

## Viewer Adaptation to Dataset APIs

The Pylon data viewer is designed to adapt to your dataset's API rather than requiring you to modify your dataset for the viewer. The viewer automatically detects and handles different data formats based on the dataset type.

### Point Cloud Registration (PCR)

The viewer expects datasets to return data in this format:

```python
# inputs format that your dataset should return
inputs = {
    'src_pc': {'pos': torch.Tensor, 'feat': torch.Tensor},
    'tgt_pc': {'pos': torch.Tensor, 'feat': torch.Tensor},
    'correspondences': torch.Tensor  # Optional
}

# labels format that your dataset should return
labels = {
    'transform': torch.Tensor  # (4, 4) transformation matrix
}
```

The viewer automatically adapts to handle:
- Different point cloud sizes
- Presence or absence of features
- Different correspondence formats
- Various transformation representations

### 2D Change Detection

```python
# inputs format that your dataset should return
inputs = {
    'img_1': torch.Tensor,  # (C, H, W) first image
    'img_2': torch.Tensor   # (C, H, W) second image
}

# labels format that your dataset should return
labels = {
    'change_map': torch.Tensor  # (H, W) binary change mask
}
```

### Semantic Segmentation

```python
# inputs format that your dataset should return
inputs = {
    'image': torch.Tensor  # (C, H, W) image
}

# labels format that your dataset should return
labels = {
    'label': torch.Tensor  # (H, W) segmentation mask
}
```

## Performance Considerations

### Large Datasets
- Large datasets may be slow in the viewer
- Point cloud LOD (Level of Detail) optimizations are automatically applied

### Point Cloud Optimization
- The viewer automatically applies LOD optimizations for large point clouds
- Default settings work well for most datasets
- Can be customized through viewer settings if needed

### Memory Management
- Viewer loads one datapoint at a time to minimize memory usage
- Automatic garbage collection between datapoint loads
- Cache settings can be adjusted for performance tuning

## Transform Handling

### Automatic Transform Application
- Viewer handles transforms automatically from your config
- Transforms are applied dynamically for visualization
- Original data remains unchanged

### Transform Handling
- Viewer applies whatever transforms are configured in your config file
- Transforms are applied according to your dataset configuration

## Advanced Features

### Custom Display Logic
- Viewer automatically detects dataset format and chooses appropriate display
- No custom display logic needed for standard formats
- Advanced datasets can provide display hints through meta_info

### Multi-Resolution Support
- Automatic Level of Detail (LOD) for large point clouds
- Density-based sampling for performance
- Interactive quality controls in viewer interface

### Dataset Navigation
- Navigate through dataset samples using the web interface
- Jump to specific indices through the interface
- Sequential browsing through the dataset

## Common Issues and Solutions

### Config File Not Found
**Problem**: Dataset doesn't appear in viewer dropdown
**Solution**: 
- Check file naming follows `{dataset_name}_data_cfg.py` pattern
- Verify file is in correct directory for dataset type
- Ensure dataset name is added to DATASET_GROUPS

### Import Errors in Config
**Problem**: Error loading dataset configuration
**Solution**:
- Add all required imports (`torch`, `data`, `utils`)
- Check dataset class name matches actual implementation
- Verify all referenced transforms/utilities exist

### Data Loading Errors
**Problem**: Dataset loads but fails to display data
**Solution**:
- Check data paths are correct and accessible
- Verify dataset parameters match actual data structure
- Test dataset loading outside viewer first

### Performance Issues
**Problem**: Viewer is slow or unresponsive
**Solution**:
- Check that your dataset implementation is efficient
- Verify data paths are correct and accessible

### Display Format Issues
**Problem**: Data appears but visualization is incorrect
**Solution**:
- Verify tensor shapes match expected formats
- Check data types (float32 for positions, int64 for labels)
- Ensure coordinate systems match viewer expectations
