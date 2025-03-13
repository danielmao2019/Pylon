"""
3DCDNet: Single-shot 3D Change Detection with Point Set Difference Modeling and Dual-path Feature Learning

This is an implementation of the 3DCDNet paper:
https://ieeexplore.ieee.org/document/9879908

Original code repository:
https://github.com/wangle53/3DCDNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any

from models.change_detection.siam3dcdnet.modules import (
    Conv1d, Conv2d, LFA, gather_neighbour
)


class C3Dnet(nn.Module):
    """Core 3D Change Detection Network.

    This is the single-branch network used within the Siamese architecture.
    """

    def __init__(self, in_d: int, out_d: int):
        """Initialize C3Dnet.

        Args:
            in_d: Input dimension (e.g., 3 for XYZ coordinates)
            out_d: Output dimension
        """
        super(C3Dnet, self).__init__()

        self.in_d = in_d
        self.out_d = out_d

        # Initial feature transformation
        self.fc0 = Conv1d(self.in_d, 64, kernel_size=1, bn=True)

        # Encoder blocks
        self.block1 = LFA(64, 128)
        self.block2 = LFA(128, 256)
        self.block3 = LFA(256, 512)
        self.block4 = LFA(512, 1024)

        # Bottleneck
        self.dt = Conv2d(1024, 1024, kernel_size=(1, 1), bn=True)

        # Decoder blocks with skip connections
        self.d4 = Conv2d(1024*2, 512, kernel_size=(1, 1), bn=True)
        self.d3 = Conv2d(512*2, 256, kernel_size=(1, 1), bn=True)
        self.d2 = Conv2d(256*2, 128, kernel_size=(1, 1), bn=True)
        self.d1 = Conv2d(128*2, 64, kernel_size=(1, 1), bn=True)

        # Final output layer
        self.d0 = Conv2d(64, self.out_d, kernel_size=(1, 1), bn=True)

    def forward(self, end_points: List):
        """Forward pass of the C3Dnet.

        Args:
            end_points: A list containing:
                - xyz: List of point coordinates at different levels
                - neigh_idx: List of neighbor indices at different levels
                - pool_idx: List of pooling indices between levels
                - unsam_idx: List of upsampling indices between levels

        Returns:
            Output feature tensor of shape (B, out_d, N, 1)
        """
        xyz, neigh_idx, pool_idx, unsam_idx = end_points

        # Encoder
        # Initial feature embedding
        out0 = self.fc0(xyz[0].permute(0, 2, 1))  # B, C, N
        out0 = out0.unsqueeze(dim=3)  # B, C, N, 1

        # Hierarchical feature extraction
        out1 = self.block1(out0, neigh_idx[0])
        out1p = self.random_sample(out1, pool_idx[0])

        out2 = self.block2(out1p, neigh_idx[1])
        out2p = self.random_sample(out2, pool_idx[1])

        out3 = self.block3(out2p, neigh_idx[2])
        out3p = self.random_sample(out3, pool_idx[2])

        out4 = self.block4(out3p, neigh_idx[3])
        out4p = self.random_sample(out4, pool_idx[3])

        # Bottleneck
        out = self.dt(out4p)

        # Decoder with skip connections
        out = torch.cat((out, out4p), 1)
        out = self.d4(out)
        out = self.nearest_interpolation(out, unsam_idx[3])

        out = torch.cat((out, out3p), 1)
        out = self.d3(out)
        out = self.nearest_interpolation(out, unsam_idx[2])

        out = torch.cat((out, out2p), 1)
        out = self.d2(out)
        out = self.nearest_interpolation(out, unsam_idx[1])

        out = torch.cat((out, out1p), 1)
        out = self.d1(out)
        out = self.nearest_interpolation(out, unsam_idx[0])

        # Final output
        out = self.d0(out)

        return out

    @staticmethod
    def random_sample(feature: torch.Tensor, pool_idx: torch.Tensor) -> torch.Tensor:
        """Sample features based on pooling indices.

        Args:
            feature: Feature tensor of shape (B, d, N, 1)
            pool_idx: Pooling indices of shape (B, N', K)

        Returns:
            Sampled features of shape (B, d, N', 1)
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        k = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints*nsamples)
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        pool_features = pool_features.reshape(batch_size, d, -1, k)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1
        return pool_features

    @staticmethod
    def nearest_interpolation(feature: torch.Tensor, interp_idx: torch.Tensor) -> torch.Tensor:
        """Interpolate features based on upsampling indices.

        Args:
            feature: Feature tensor of shape (B, d, N, 1)
            interp_idx: Interpolation indices of shape (B, up_num_points, 1)

        Returns:
            Interpolated features of shape (B, d, up_num_points, 1)
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(feature, 2, interp_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        interpolated_features = interpolated_features.unsqueeze(3)  # batch*channel*npoints*1
        return interpolated_features


class Siam3DCDNet(nn.Module):
    """
    3DCDNet: Single-shot 3D Change Detection with Point Set Difference Modeling and Dual-path Feature Learning.

    This model uses a dual-path architecture with point set difference modeling
    for detecting changes between two point clouds.
    """

    def __init__(
        self,
        num_classes: int = 2,
        input_dim: int = 3,
        feature_dims: List[int] = [64, 128, 256],
        dropout: float = 0.1,
        k_neighbors: int = 16,
        sub_sampling_ratio: List[int] = [4, 4, 4, 4]
    ):
        """Initialize Siam3DCDNet model.

        Args:
            num_classes: Number of output classes
            input_dim: Input dimension (e.g., 3 for XYZ, more if RGB included)
            feature_dims: List of feature dimensions for hierarchical levels
            dropout: Dropout rate
            k_neighbors: Number of neighbors for feature aggregation
            sub_sampling_ratio: Downsampling ratio for each level
        """
        super(Siam3DCDNet, self).__init__()

        self.num_classes = num_classes
        self.input_dim = input_dim
        self.feature_dims = feature_dims
        self.k_neighbors = k_neighbors
        self.sub_sampling_ratio = sub_sampling_ratio

        # Main feature extractor network (shared weights)
        self.net = C3Dnet(input_dim, 64)

        # Change detection head
        self.mlp1 = Conv1d(64, 32, kernel_size=1, bn=True)
        self.mlp2 = Conv1d(32, num_classes, kernel_size=1, bias=False, bn=False, activation=None)

    @staticmethod
    def nearest_feature_difference(raw: torch.Tensor, query: torch.Tensor,
                                   nearest_idx: torch.Tensor) -> torch.Tensor:
        """Compute feature differences between point clouds.

        Args:
            raw: Features from first point cloud, shape (B, C, N, 1)
            query: Features from second point cloud, shape (B, C, N, 1)
            nearest_idx: K-nearest neighbors indices, shape (B, N, K)

        Returns:
            Fused difference features of shape (B, C, N, 1)
        """
        nearest_features = gather_neighbour(query, nearest_idx)
        fused_features = torch.mean(torch.abs(raw - nearest_features), -1)
        return fused_features

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass of the Siam3DCDNet model.

        Args:
            data_dict: Dictionary containing:
                - pc_0: First point cloud data dict with:
                    - xyz: List of point coordinates at each level
                    - neighbors_idx: List of neighbor indices at each level
                    - pool_idx: List of pooling indices between levels
                    - unsam_idx: List of upsampling indices between levels
                - pc_1: Second point cloud with the same structure
                - knearst_idx_in_another_pc: Cross-point cloud KNN indices

        Returns:
            Dictionary with model outputs:
                - logits_0: Logits for the first point cloud
                - logits_1: Logits for the second point cloud
        """
        # Extract data from input dictionary
        pc_0 = data_dict['pc_0']
        pc_1 = data_dict['pc_1']

        # Extract point cloud components for each point cloud
        end_points0 = [
            pc_0['xyz'],
            pc_0['neighbors_idx'],
            pc_0['pool_idx'],
            pc_0['unsam_idx']
        ]

        end_points1 = [
            pc_1['xyz'],
            pc_1['neighbors_idx'],
            pc_1['pool_idx'],
            pc_1['unsam_idx']
        ]

        # Get cross-cloud KNN indices
        knearest_idx = [
            pc_0['knearst_idx_in_another_pc'],
            pc_1['knearst_idx_in_another_pc']
        ]

        # Forward pass through the network for each point cloud
        out0 = self.net(end_points0)  # B, C, N, 1
        out1 = self.net(end_points1)  # B, C, N, 1

        # Extract KNN indices
        knearest_01, knearest_10 = knearest_idx

        # Compute feature differences
        fout0 = self.nearest_feature_difference(out0, out1, knearest_01)
        fout1 = self.nearest_feature_difference(out1, out0, knearest_10)

        # Process point cloud 0 features
        fout0 = self.mlp1(fout0.squeeze(-1))  # B, C, N
        fout1 = self.mlp1(fout1.squeeze(-1))  # B, C, N

        fout0 = self.mlp2(fout0)  # B, nc, N
        fout1 = self.mlp2(fout1)  # B, nc, N

        # Return logits as dictionary (instead of log_softmax as in original) for compatibility
        # with Pylon's criteria and metrics that expect raw logits
        logits_0 = fout0.transpose(2, 1)  # B, N, nc
        logits_1 = fout1.transpose(2, 1)  # B, N, nc

        return {
            'logits_0': logits_0,
            'logits_1': logits_1
        }


# Factory function for creating model
def get_model(num_classes: int = 2, **kwargs) -> Siam3DCDNet:
    """Factory function for creating 3DCDNet model.

    Args:
        num_classes: Number of output classes
        **kwargs: Additional arguments to pass to model constructor

    Returns:
        Siam3DCDNet model instance
    """
    return Siam3DCDNet(num_classes=num_classes, **kwargs) 