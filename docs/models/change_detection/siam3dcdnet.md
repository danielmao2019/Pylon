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
import torch

from data.structures.three_d.point_cloud.point_cloud import PointCloud
from models.change_detection import Siam3DCDNet

# Create model
model = Siam3DCDNet(
    num_classes=2,  # Binary change detection
    input_dim=3,    # XYZ coordinates
    feature_dims=[64, 128, 256],
    dropout=0.1,
    k_neighbors=16
)

# Build the inputs using the PointCloud API. Each PointCloud already validates
# that `xyz` is float, two-dimensional, and non-empty, and exposes `.num_points`.
batch_size = 2
num_points = 1024
xyz0 = torch.randn(batch_size, num_points, 3, dtype=torch.float32)
xyz1 = torch.randn(batch_size, num_points, 3, dtype=torch.float32)
pc_0 = PointCloud(xyz=xyz0)
pc_1 = PointCloud(xyz=xyz1)

# The distributed collator will augment each PointCloud with the neighbors/pooling
# indices that the backbone expects. At inference time, assemble a mapping of the
# two PointCloud objects and pass it to the network.
inputs = {'pc_0': pc_0, 'pc_1': pc_1}
outputs = model(inputs)

The collator now produces fully validated `PointCloud` instances whose attributes
carry the pooled/neighbor indices that the backbone consumes. Treat these objects
as the canonical samples and pass them through the mapping; do not fall back to the
old dict-of-tensors representation. Because the `PointCloud` constructor already
asserts a `(N, 3)` float tensor, downstream code can simply rely on `.num_points`.

# Output format
# {
#     'logits_0': tensor of shape (B, N, num_classes),  # Logits for first point cloud
#     'logits_1': tensor of shape (B, N, num_classes)   # Logits for second point cloud
# }
```

Point clouds should be referenced through `pc.num_points` rather than inspecting `pc.xyz.shape[0]`.

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
