"""
Core modules for the 3DCDNet model implementation.

This file includes the LocalFeatureAggregation (LFA) module and its components,
as described in the original 3DCDNet paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class MLP(nn.Module):
    """Simple multi-layer perceptron module."""
    
    def __init__(self, in_dim: int, hidden_dims: List[int], bn: bool = True, 
                 dropout: Optional[float] = None, activation: Optional[str] = "relu"):
        """Initialize MLP.
        
        Args:
            in_dim: Input dimension
            hidden_dims: List of hidden dimensions
            bn: Whether to use batch normalization
            dropout: Dropout rate (if None, no dropout)
            activation: Activation function to use
        """
        super(MLP, self).__init__()
        
        self.layers = nn.ModuleList()
        dims = [in_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            
            if bn and i < len(dims) - 2:
                self.layers.append(nn.BatchNorm1d(dims[i+1]))
                
            if activation and i < len(dims) - 2:
                if activation == "relu":
                    self.layers.append(nn.ReLU())
                elif activation == "leaky_relu":
                    self.layers.append(nn.LeakyReLU(0.1))
                
            if dropout and i < len(dims) - 2:
                self.layers.append(nn.Dropout(dropout))
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, N, in_dim)
            
        Returns:
            Output tensor of shape (B, N, hidden_dims[-1])
        """
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm1d):
                # BatchNorm1d expects (B, C, N) format
                shape = x.shape
                x = layer(x.transpose(1, 2)).transpose(1, 2)
            else:
                x = layer(x)
        return x


class Conv1d(nn.Module):
    """1D convolution with batch normalization and activation."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        bn: bool = True,
        activation: nn.Module = nn.ReLU(inplace=True),
        bias: bool = True
    ):
        """Initialize Conv1d.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolution kernel
            stride: Stride of the convolution
            padding: Zero-padding added to both sides of the input
            bn: Whether to use batch normalization
            activation: Activation function to use
            bias: Whether to use bias in the convolution
        """
        super(Conv1d, self).__init__()
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        
        self.bn = nn.BatchNorm1d(out_channels) if bn else None
        self.activation = activation
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, N)
            
        Returns:
            Output tensor of shape (B, C, N)
        """
        x = self.conv(x)
        
        if self.bn is not None:
            x = self.bn(x)
            
        if self.activation is not None:
            x = self.activation(x)
            
        return x


class Conv2d(nn.Module):
    """2D convolution with batch normalization and activation."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: Tuple[int, int] = (1, 1),
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        bn: bool = True,
        activation: nn.Module = nn.ReLU(inplace=True),
        bias: bool = True
    ):
        """Initialize Conv2d.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolution kernel
            stride: Stride of the convolution
            padding: Zero-padding added to both sides of the input
            bn: Whether to use batch normalization
            activation: Activation function to use
            bias: Whether to use bias in the convolution
        """
        super(Conv2d, self).__init__()
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.activation = activation
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        x = self.conv(x)
        
        if self.bn is not None:
            x = self.bn(x)
            
        if self.activation is not None:
            x = self.activation(x)
            
        return x


def gather_neighbour(feature: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    """Gather features of neighboring points.
    
    Args:
        feature: Feature tensor of shape (B, C, N, 1)
        neighbor_idx: Indices of neighbors of shape (B, N, K)
        
    Returns:
        Gathered features of shape (B, C, N, K)
    """
    feature = feature.squeeze(dim=3).transpose(2, 1)  # B, N, C
    batch_size, num_points, num_features = feature.shape
    k_neighbors = neighbor_idx.shape[2]
    
    # Reshape neighbor_idx for efficient gathering
    index_input = neighbor_idx.reshape(batch_size, -1)  # B, N*K
    
    # Gather features from all points
    features = torch.gather(
        feature, 
        1, 
        index_input.unsqueeze(-1).repeat(1, 1, num_features)  # B, N*K, C
    )
    
    # Reshape back to original dimensions
    features = features.reshape(batch_size, num_points, k_neighbors, num_features)  # B, N, K, C
    features = features.permute(0, 3, 1, 2)  # B, C, N, K
    
    return features


class SpatialPointsEncoding(nn.Module):
    """Spatial Points Encoding (SPE) module from 3DCDNet."""
    
    def __init__(self, d_in: int, d_out: int):
        """Initialize SPE module.
        
        Args:
            d_in: Input dimension
            d_out: Output dimension
        """
        super(SpatialPointsEncoding, self).__init__()
        
        self.mlp = Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)
        
    def forward(self, feature: torch.Tensor, neigh_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            feature: Feature tensor of shape (B, C, N, 1)
            neigh_idx: Indices of neighbors of shape (B, N, K)
            
        Returns:
            Encoded features of shape (B, C_out, N, 1)
        """
        # Gather features of neighboring points
        f_neigh = gather_neighbour(feature, neigh_idx)  # B, C, N, K
        
        # Concatenate central features with neighboring features
        f_neigh = torch.cat((feature, f_neigh), dim=3)  # B, C, N, K+1
        
        # Apply MLP
        f_agg = self.mlp(f_neigh)  # B, C_out, N, K+1
        
        # Sum over neighbors to get invariant features
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)  # B, C_out, N, 1
        
        return f_agg


class LocalFeatureExtraction(nn.Module):
    """Local Feature Extraction (LFE) module from 3DCDNet."""
    
    def __init__(self, d_in: int, d_out: int):
        """Initialize LFE module.
        
        Args:
            d_in: Input dimension
            d_out: Output dimension
        """
        super(LocalFeatureExtraction, self).__init__()
        
        self.mlp1 = Conv2d(d_in, d_in, kernel_size=(1, 1), bn=True)
        self.mlp2 = Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)
        self.mlp3 = Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)
        
    def forward(self, feature: torch.Tensor, neigh_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            feature: Feature tensor of shape (B, C, N, 1)
            neigh_idx: Indices of neighbors of shape (B, N, K)
            
        Returns:
            Extracted features of shape (B, C_out, N, 1)
        """
        # Gather features of neighboring points
        f_neigh = gather_neighbour(feature, neigh_idx)  # B, C, N, K
        
        # Apply first MLP
        f_neigh = self.mlp1(f_neigh)  # B, C, N, K
        
        # Sum over neighbors to get invariant features
        f_neigh = torch.sum(f_neigh, dim=3, keepdim=True)  # B, C, N, 1
        
        # Apply second MLP
        f_neigh = self.mlp2(f_neigh)  # B, C_out, N, 1
        
        # Apply third MLP on central features
        feature = self.mlp3(feature)  # B, C_out, N, 1
        
        # Combine features
        f_agg = f_neigh + feature  # B, C_out, N, 1
        
        return f_agg


class LocalFeatureAggregation(nn.Module):
    """Local Feature Aggregation (LFA) module from 3DCDNet."""
    
    def __init__(self, d_in: int, d_out: int):
        """Initialize LFA module.
        
        Args:
            d_in: Input dimension
            d_out: Output dimension
        """
        super(LocalFeatureAggregation, self).__init__()
        
        self.spe = SpatialPointsEncoding(d_in, d_out)
        self.lfe = LocalFeatureExtraction(d_in, d_out)
        self.mlp = Conv2d(d_out, d_out, kernel_size=(1, 1), bn=True)
        
    def forward(self, feature: torch.Tensor, neigh_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            feature: Feature tensor of shape (B, C, N, 1)
            neigh_idx: Indices of neighbors of shape (B, N, K)
            
        Returns:
            Aggregated features of shape (B, C_out, N, 1)
        """
        # Apply SPE
        spe_features = self.spe(feature, neigh_idx)  # B, C_out, N, 1
        
        # Apply LFE
        lfe_features = self.lfe(feature, neigh_idx)  # B, C_out, N, 1
        
        # Combine features
        combined_features = spe_features + lfe_features  # B, C_out, N, 1
        
        # Apply final MLP
        f_agg = self.mlp(combined_features)  # B, C_out, N, 1
        
        return f_agg


class PointSetDifferenceModule(nn.Module):
    """Point Set Difference Module as described in the 3DCDNet paper."""
    
    def __init__(self, feature_dim: int, knn_size: int = 16):
        """Initialize Point Set Difference Module.
        
        Args:
            feature_dim: Dimension of input features
            knn_size: Number of nearest neighbors for cross-point cloud search
        """
        super(PointSetDifferenceModule, self).__init__()
        
        self.knn_size = knn_size
        
        # MLP for feature difference processing
        self.diff_mlp = MLP(feature_dim, [feature_dim, feature_dim])
        
        # MLP for feature similarity processing
        self.sim_mlp = MLP(feature_dim, [feature_dim, feature_dim])
        
        # Final MLP to combine difference and similarity
        self.final_mlp = MLP(feature_dim * 2, [feature_dim * 2, feature_dim])
    
    def forward(self, features_0: torch.Tensor, features_1: torch.Tensor, 
                knn_idx_0_to_1: torch.Tensor, knn_idx_1_to_0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the Point Set Difference Module.
        
        Args:
            features_0: Features from first point cloud, shape (B, N, C)
            features_1: Features from second point cloud, shape (B, N, C)
            knn_idx_0_to_1: KNN indices from pc0 to pc1, shape (B, N, K)
            knn_idx_1_to_0: KNN indices from pc1 to pc0, shape (B, N, K)
            
        Returns:
            Tuple containing:
            - Difference features for pc0, shape (B, N, C)
            - Difference features for pc1, shape (B, N, C)
        """
        batch_size, num_points, feature_dim = features_0.size()
        
        # Extract neighboring features using KNN indices
        batch_indices = torch.arange(batch_size, device=features_0.device).view(-1, 1, 1).repeat(1, num_points, self.knn_size)
        
        # Get neighbors from pc1 for each point in pc0
        neighbors_1_for_0 = features_1[batch_indices, knn_idx_0_to_1]  # (B, N, K, C)
        
        # Get neighbors from pc0 for each point in pc1
        neighbors_0_for_1 = features_0[batch_indices, knn_idx_1_to_0]  # (B, N, K, C)
        
        # Calculate feature differences
        features_0_expanded = features_0.unsqueeze(2).repeat(1, 1, self.knn_size, 1)  # (B, N, K, C)
        features_1_expanded = features_1.unsqueeze(2).repeat(1, 1, self.knn_size, 1)  # (B, N, K, C)
        
        # Compute differences
        diff_0_to_1 = features_0_expanded - neighbors_1_for_0  # (B, N, K, C)
        diff_1_to_0 = features_1_expanded - neighbors_0_for_1  # (B, N, K, C)
        
        # Process differences with MLP
        diff_0_to_1 = diff_0_to_1.reshape(batch_size * num_points, self.knn_size, feature_dim)
        diff_1_to_0 = diff_1_to_0.reshape(batch_size * num_points, self.knn_size, feature_dim)
        
        diff_0_to_1 = self.diff_mlp(diff_0_to_1)  # (B*N, K, C)
        diff_1_to_0 = self.diff_mlp(diff_1_to_0)  # (B*N, K, C)
        
        # Reshape back
        diff_0_to_1 = diff_0_to_1.reshape(batch_size, num_points, self.knn_size, feature_dim)
        diff_1_to_0 = diff_1_to_0.reshape(batch_size, num_points, self.knn_size, feature_dim)
        
        # Max pooling over K neighbors
        diff_0_to_1 = torch.max(diff_0_to_1, dim=2)[0]  # (B, N, C)
        diff_1_to_0 = torch.max(diff_1_to_0, dim=2)[0]  # (B, N, C)
        
        # Compute similarities (dot product)
        sim_0_to_1 = torch.sum(features_0_expanded * neighbors_1_for_0, dim=-1, keepdim=True)  # (B, N, K, 1)
        sim_1_to_0 = torch.sum(features_1_expanded * neighbors_0_for_1, dim=-1, keepdim=True)  # (B, N, K, 1)
        
        # Normalize similarities
        sim_0_to_1 = F.softmax(sim_0_to_1, dim=2)  # (B, N, K, 1)
        sim_1_to_0 = F.softmax(sim_1_to_0, dim=2)  # (B, N, K, 1)
        
        # Weighted aggregation of neighbor features
        weighted_neighbors_1_for_0 = torch.sum(neighbors_1_for_0 * sim_0_to_1, dim=2)  # (B, N, C)
        weighted_neighbors_0_for_1 = torch.sum(neighbors_0_for_1 * sim_1_to_0, dim=2)  # (B, N, C)
        
        # Process similarity features
        sim_0_to_1 = self.sim_mlp(weighted_neighbors_1_for_0)  # (B, N, C)
        sim_1_to_0 = self.sim_mlp(weighted_neighbors_0_for_1)  # (B, N, C)
        
        # Combine difference and similarity features
        combined_0_to_1 = torch.cat([diff_0_to_1, sim_0_to_1], dim=-1)  # (B, N, 2C)
        combined_1_to_0 = torch.cat([diff_1_to_0, sim_1_to_0], dim=-1)  # (B, N, 2C)
        
        # Final processing
        diff_features_0 = self.final_mlp(combined_0_to_1)  # (B, N, C)
        diff_features_1 = self.final_mlp(combined_1_to_0)  # (B, N, C)
        
        return diff_features_0, diff_features_1
