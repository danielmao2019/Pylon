# 3DCDNet: Single-shot 3D Change Detection

This is an implementation of the model proposed in the paper:

**3DCDNet: Single-shot 3D Change Detection with Point Set Difference Modeling and Dual-path Feature Learning**  
*Yongtao Chen, Linlin Ge, Ruizhi Chen, Xiao Huang, and Guo Zhang*  
IEEE Transactions on Geoscience and Remote Sensing, 2022  
[Paper Link](https://ieeexplore.ieee.org/document/9879908)

## Model Architecture

3DCDNet is designed for change detection in 3D point clouds. The model architecture consists of:

1. **Dual-path Encoder**: Processes two point clouds using hierarchical feature extraction with shared weights.
2. **Feature Difference Computation**: Captures changes between the two point clouds using nearest neighbors.
3. **Change Classification**: Produces the final change detection results for each point cloud.

## Key Components

### C3Dnet

The core 3D change detection network which extracts hierarchical features from point clouds using Local Feature Aggregation (LFA) modules.

### Local Feature Aggregation (LFA)

This module combines:
- Spatial Points Encoding (SPE) for capturing spatial relationships
- Local Feature Extraction (LFE) for learning local patterns

### Feature Difference Computation

The original implementation computes feature differences using nearest neighbor information:
```python
nearest_features = gather_neighbour(query, nearest_idx)
fused_features = torch.mean(torch.abs(raw - nearest_features), -1)
```

## Usage

```python
from models.change_detection import Siam3DCDNet

# Create model
model = Siam3DCDNet(
    num_classes=2,  # Binary change detection
    input_dim=3,    # XYZ coordinates
    feature_dims=[64, 128, 256],
    dropout=0.1,
    k_neighbors=16
)

# Input format for the model
# data_dict = {
#     'pc_0': {
#         'xyz': [xyz0_l0, xyz0_l1, ...],  # Point coordinates at each level
#         'neighbors_idx': [neigh0_l0, ...],  # Neighbor indices at each level
#         'pool_idx': [pool0_l0_l1, ...],  # Pooling indices between levels
#         'unsam_idx': [unsam0_l0_l1, ...],  # Upsampling indices between levels
#         'knearst_idx_in_another_pc': knn0_to_1  # Cross-cloud KNN indices
#     },
#     'pc_1': {
#         # Same structure as pc_0
#     }
# }

# Output format
# {
#     'logits_0': tensor of shape (B, N, num_classes),  # Logits for first point cloud
#     'logits_1': tensor of shape (B, N, num_classes)   # Logits for second point cloud
# }
```

## Citation

```
@ARTICLE{chen20223dcdnet,
  author={Chen, Yongtao and Ge, Linlin and Chen, Ruizhi and Huang, Xiao and Zhang, Guo},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={3DCDNet: Single-Shot 3D Change Detection With Point Set Difference Modeling and Dual-Path Feature Learning}, 
  year={2022},
  volume={60},
  pages={1-15},
  doi={10.1109/TGRS.2022.3196927}
}
``` 