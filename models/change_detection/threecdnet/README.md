# 3DCDNet: Single-shot 3D Change Detection

This is an implementation of the paper "3DCDNet: Single-shot 3D Change Detection with Point Set Difference Modeling and Dual-path Feature Learning" (IEEE Transactions on Geoscience and Remote Sensing, 2022).

## Paper Details

- **Title**: 3DCDNet: Single-shot 3D Change Detection with Point Set Difference Modeling and Dual-path Feature Learning
- **Authors**: Yongtao Chen, Linlin Ge, Ruizhi Chen, Xiao Huang, and Guo Zhang
- **Publication**: IEEE Transactions on Geoscience and Remote Sensing, 2022
- **DOI**: [10.1109/TGRS.2022.3203769](https://doi.org/10.1109/TGRS.2022.3203769)
- **Paper Link**: [IEEE Xplore](https://ieeexplore.ieee.org/document/9879908)
- **Original Code**: [GitHub](https://github.com/PointCloudYC/3DCDNet)

## Model Architecture

The 3DCDNet model consists of three main components:

1. **Dual-path Encoders**: Two independent encoders process each point cloud separately, extracting hierarchical features at multiple resolutions.

2. **Point Set Difference Module (PSDM)**: This module captures changes between point clouds by:
   - Computing feature differences between corresponding points
   - Calculating feature similarities across point clouds
   - Combining difference and similarity information

3. **Feature Decoder & Classification Head**: The decoder upsamples features from coarser to finer resolutions, and the classification head predicts change labels.

![3DCDNet Architecture](https://github.com/PointCloudYC/3DCDNet/raw/main/images/architecture.jpg)

## Usage

### Loading the Model

```python
from models.change_detection import ThreeCDNet

# Create model instance
model = ThreeCDNet(
    num_classes=2,          # Binary classification (change/no change)
    input_dim=3,            # XYZ coordinates only (increase if RGB included)
    feature_dims=[64, 128, 256],  # Feature dimensions at each level
    dropout=0.1             # Dropout rate
)
```

### Input Format

The model expects a specific input format structured as a dictionary with two point clouds:

```python
data_dict = {
    'pc_0': {  # First point cloud (t0)
        'xyz': [xyz_0_l0, xyz_0_l1, ...],  # List of coordinates at each level
        'feat': [feat_0],  # Optional features (RGB, etc.)
        'neighbors_idx': [neighbors_0_l0, neighbors_0_l1, ...],  # KNN indices
        'pool_idx': [pool_0_l0_l1, ...],  # Pooling indices between levels
        'knearst_idx_in_another_pc': k_nearest_0_to_1  # Cross-cloud KNN indices
    },
    'pc_1': {  # Second point cloud (t1) with same structure
        # Similar structure as pc_0
    }
}
```

### Training

Use the provided configuration file for training:

```bash
python train.py --config models/change_detection/threecdnet/config.yml
```

## Dataset Compatibility

The model is designed to work with the SLPCCD (Street-Level Point Cloud Change Detection) dataset. It can also be adapted to work with other point cloud change detection datasets by ensuring the same input format is provided.

## References

Please cite the original paper if you use this implementation:

```
@ARTICLE{3DCDNet2022,
  author={Chen, Yongtao and Ge, Linlin and Chen, Ruizhi and Huang, Xiao and Zhang, Guo},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={3DCDNet: Single-Shot 3D Change Detection With Point Set Difference Modeling and Dual-Path Feature Learning}, 
  year={2022},
  volume={60},
  pages={1-15},
  doi={10.1109/TGRS.2022.3203769}
}
``` 