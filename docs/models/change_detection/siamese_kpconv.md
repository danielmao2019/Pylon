# SiameseKPConv for 3D Point Cloud Change Detection

This module provides a standalone implementation of SiameseKPConv for 3D point cloud change detection, based on the original [torch-points3d-SiameseKPConv](https://github.com/humanpose1/torch-points3d-SiameseKPConv) repository but without dependencies on the torch_points3d framework.

## Architecture

SiameseKPConv is a siamese network based on KPConv (Kernel Point Convolution) for detecting changes between two point clouds of the same area captured at different times:

1. **Encoder**: Both point clouds are processed through the same KPConv-based encoder to extract features
2. **Feature Differencing**: Corresponding points between the two point clouds are found using kNN, and their features are subtracted
3. **Decoder**: The difference features are processed through a decoder with skip connections
4. **Classification**: A final MLP classifies each point as changed or unchanged

## Implementation Details

The implementation includes:

- **KPConv operator**: Defines convolution kernels as sets of points in space
- **SimpleBlock**: Basic KPConv block with convolution -> batch norm -> activation
- **ResnetBBlock**: Bottleneck Resnet block for KPConv with residual connections
- **KPDualBlock**: Combines multiple blocks for more complex processing

## Usage

```python
from data.structures.three_d.point_cloud.point_cloud import PointCloud
from models.change_detection.siamese_kpconv.siamese_kpconv_model import SiameseKPConv

# Initialize model
model = SiameseKPConv(
    in_channels=3,            # Number of input features per point
    out_channels=2,           # Number of output classes
    point_influence=0.1,      # Radius of influence for each kernel point
    down_channels=[32, 64],   # Feature dimensions for encoder layers
    up_channels=[64, 32],     # Feature dimensions for decoder layers
    conv_type='simple'        # Type of convolution blocks
)

# Input format (two PointCloud objects)
inputs = {
    'pc_0': PointCloud(
        xyz=pos_tensor_1,            # [N, 3] point positions
        data={'x': feat_tensor_1, 'batch': batch_tensor_1}
    ),
    'pc_1': PointCloud(
        xyz=pos_tensor_2,
        data={'x': feat_tensor_2, 'batch': batch_tensor_2}
    )
}

# Forward pass
change_logits = model(inputs)  # [N, num_classes] - raw logits, not softmax probabilities
```

`PointCloud` enforces the `(N, 3)` coordinate shape, non-empty tensors, and consistent optional fields, so you no longer need to inspect `.xyz.shape[0]` for point counts. Rely on `pc.num_points` when deriving batch sizes or pooling operations, and keep any extra `data` fields synchronized with that value.

## References

This implementation is based on the original papers:

- Thomas, H., Qi, C. R., Deschaud, J. E., Marcotegui, B., Goulette, F., & Guibas, L. J. (2019). *KPConv: Flexible and deformable convolution for point clouds.* In Proceedings of the IEEE/CVF International Conference on Computer Vision.
  
- Hatem, H., Villalón, F., Arteaga, A., & El-Sayed, E. (2022). *SiameseKPConv: A Siamese KPConv Network Architecture for 3D Point Cloud Change Detection.* IEEE Access.
