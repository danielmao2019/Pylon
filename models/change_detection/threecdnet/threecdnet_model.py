"""
3DCDNet: Single-shot 3D Change Detection with Point Set Difference Modeling and Dual-path Feature Learning

This is an implementation of the 3DCDNet paper:
https://ieeexplore.ieee.org/document/9879908

Original code repository:
https://github.com/PointCloudYC/3DCDNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any


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


class PointNetSAModule(nn.Module):
    """PointNet Set Abstraction Module."""
    
    def __init__(self, mlp_dims: List[int], bn: bool = True):
        """Initialize PointNet SA module.
        
        Args:
            mlp_dims: List of dimensions for MLP
            bn: Whether to use batch normalization
        """
        super(PointNetSAModule, self).__init__()
        
        self.mlp = MLP(mlp_dims[0], mlp_dims[1:], bn=bn)
    
    def forward(self, xyz: torch.Tensor, features: torch.Tensor, neighbors_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            xyz: Point coordinates, shape (B, N, 3)
            features: Point features, shape (B, N, C)
            neighbors_idx: K-nearest neighbors indices, shape (B, N, K)
            
        Returns:
            Updated features of shape (B, N, mlp_dims[-1])
        """
        batch_size, num_points, k = neighbors_idx.size()
        
        # Get features of neighbors
        batch_indices = torch.arange(batch_size, device=xyz.device).view(-1, 1, 1).repeat(1, num_points, k)
        point_indices = neighbors_idx
        
        # Gather features for each point's neighbors
        features_neighbors = features[batch_indices, point_indices]  # (B, N, K, C)
        
        # Gather coordinates for each point's neighbors
        xyz_neighbors = xyz[batch_indices, point_indices]  # (B, N, K, 3)
        
        # Calculate relative positions
        xyz_central = xyz.unsqueeze(2).repeat(1, 1, k, 1)  # (B, N, K, 3)
        xyz_local = xyz_neighbors - xyz_central  # (B, N, K, 3)
        
        # Concatenate local position with features
        if features is not None:
            features_central = features.unsqueeze(2).repeat(1, 1, k, 1)  # (B, N, K, C)
            features_local = torch.cat([xyz_local, features_central, features_neighbors], dim=-1)  # (B, N, K, 3+C+C)
        else:
            features_local = xyz_local  # (B, N, K, 3)
        
        # Reshape for MLP
        features_local = features_local.view(batch_size * num_points, k, -1)
        
        # Apply MLP
        features_local = self.mlp(features_local)  # (B*N, K, mlp_dims[-1])
        
        # Max pooling to get invariant features
        features_local = features_local.view(batch_size, num_points, k, -1)
        new_features = torch.max(features_local, dim=2)[0]  # (B, N, mlp_dims[-1])
        
        return new_features


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


class HierarchicalFeatureExtractor(nn.Module):
    """Hierarchical feature extractor for point clouds."""
    
    def __init__(self, feature_dims: List[int], input_dim: int = 3):
        """Initialize hierarchical feature extractor.
        
        Args:
            feature_dims: List of feature dimensions for each level
            input_dim: Input dimension (e.g., 3 for XYZ, more if features included)
        """
        super(HierarchicalFeatureExtractor, self).__init__()
        
        self.num_levels = len(feature_dims)
        
        # Create SA modules for each level
        self.sa_modules = nn.ModuleList()
        
        # First level
        self.sa_modules.append(PointNetSAModule([input_dim, 32, 32, feature_dims[0]]))
        
        # Subsequent levels
        for i in range(1, self.num_levels):
            self.sa_modules.append(
                PointNetSAModule([feature_dims[i-1], feature_dims[i-1], feature_dims[i]])
            )
    
    def forward(self, xyz_list: List[torch.Tensor], features: Optional[torch.Tensor],
                neighbors_idx_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass of the hierarchical feature extractor.
        
        Args:
            xyz_list: List of point coordinates at each level, each of shape (B, N_l, 3)
            features: Initial point features, shape (B, N_0, C) or None
            neighbors_idx_list: List of K-nearest neighbors indices at each level
            
        Returns:
            List of features at each level
        """
        features_list = []
        
        # First level
        current_features = self.sa_modules[0](xyz_list[0], features, neighbors_idx_list[0])
        features_list.append(current_features)
        
        # Subsequent levels
        for i in range(1, self.num_levels):
            current_features = self.sa_modules[i](xyz_list[i], current_features, neighbors_idx_list[i])
            features_list.append(current_features)
        
        return features_list


class FeatureDecoder(nn.Module):
    """Feature decoder module for upsampling feature hierarchies."""
    
    def __init__(self, feature_dims: List[int]):
        """Initialize feature decoder.
        
        Args:
            feature_dims: List of feature dimensions at each level (in descending order)
        """
        super(FeatureDecoder, self).__init__()
        
        self.num_levels = len(feature_dims)
        
        # Create upsampling MLPs
        self.up_mlps = nn.ModuleList()
        
        # MLPs for skip connections
        self.skip_mlps = nn.ModuleList()
        
        # Create MLPs for each level except the last one
        for i in range(self.num_levels - 1):
            # Upsampling MLP
            self.up_mlps.append(
                MLP(feature_dims[i+1], [feature_dims[i+1], feature_dims[i]])
            )
            
            # Skip connection MLP
            self.skip_mlps.append(
                MLP(feature_dims[i] * 2, [feature_dims[i] * 2, feature_dims[i]])
            )
    
    def forward(self, features_list: List[torch.Tensor], 
                pool_idx_list: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass of the feature decoder.
        
        Args:
            features_list: List of features at each level, each of shape (B, N_l, C_l)
            pool_idx_list: List of pooling indices for upsampling
            
        Returns:
            Decoded features at the finest level
        """
        features = features_list[-1]  # Start with coarsest level
        
        # Upsample from coarsest to finest
        for i in range(self.num_levels - 2, -1, -1):
            batch_size = features.size(0)
            num_points = features_list[i].size(1)
            
            # Apply MLP before upsampling
            features = self.up_mlps[i](features)
            
            # Upsample using pooling indices
            features_upsampled = self._upsample_using_pool_indices(
                features, pool_idx_list[i], num_points
            )
            
            # Skip connection
            features = torch.cat([features_list[i], features_upsampled], dim=-1)
            features = self.skip_mlps[i](features)
        
        return features
    
    def _upsample_using_pool_indices(self, features: torch.Tensor,
                                      pool_indices: torch.Tensor,
                                      target_size: int) -> torch.Tensor:
        """Upsample features using pooling indices.
        
        Args:
            features: Features to upsample, shape (B, N_coarse, C)
            pool_indices: Pooling indices, shape (B, N_fine)
            target_size: Target number of points
            
        Returns:
            Upsampled features of shape (B, N_fine, C)
        """
        batch_size, _, feature_dim = features.size()
        
        # Create batch indices
        batch_indices = torch.arange(batch_size, device=features.device).view(-1, 1).repeat(1, target_size)
        
        # Gather features using pooling indices
        upsampled_features = features[batch_indices, pool_indices]
        
        return upsampled_features


class ThreeCDNet(nn.Module):
    """
    3DCDNet model for 3D point cloud change detection.
    
    This model uses a dual-path architecture with point set difference modeling
    for detecting changes between two point clouds.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        input_dim: int = 3,
        feature_dims: List[int] = [64, 128, 256],
        dropout: float = 0.1
    ):
        """Initialize 3DCDNet model.
        
        Args:
            num_classes: Number of output classes
            input_dim: Input dimension (e.g., 3 for XYZ, more if RGB included)
            feature_dims: List of feature dimensions for hierarchical levels
            dropout: Dropout rate
        """
        super(ThreeCDNet, self).__init__()
        
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.feature_dims = feature_dims
        
        # Dual-path encoders
        self.encoder_0 = HierarchicalFeatureExtractor(feature_dims, input_dim)
        self.encoder_1 = HierarchicalFeatureExtractor(feature_dims, input_dim)
        
        # Point Set Difference Module for each level
        self.psdm_modules = nn.ModuleList([
            PointSetDifferenceModule(dim) for dim in feature_dims
        ])
        
        # Decoders
        self.decoder_0 = FeatureDecoder(feature_dims)
        self.decoder_1 = FeatureDecoder(feature_dims)
        
        # Final change detection heads
        self.change_head = nn.Sequential(
            nn.Linear(feature_dims[0] * 2, feature_dims[0]),
            nn.BatchNorm1d(feature_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dims[0], num_classes)
        )
    
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass of the 3DCDNet model.
        
        Args:
            data_dict: Dictionary containing:
                - pc_0: First point cloud data dict with:
                    - xyz: List of point coordinates at each level
                    - feat: Point features (optional)
                    - neighbors_idx: List of neighbor indices at each level
                    - pool_idx: List of pooling indices between levels
                - pc_1: Second point cloud with the same structure
                - knearst_idx_in_another_pc: Cross-point cloud KNN indices
                
        Returns:
            Dictionary with model outputs:
                - logits: Change classification logits
        """
        # Extract inputs
        pc_0 = data_dict['pc_0']
        pc_1 = data_dict['pc_1']
        
        xyz_0_list = pc_0['xyz']
        xyz_1_list = pc_1['xyz']
        
        feat_0 = pc_0.get('feat', [None])[0]  # Get features if available, otherwise None
        feat_1 = pc_1.get('feat', [None])[0]
        
        neighbors_idx_0_list = pc_0['neighbors_idx']
        neighbors_idx_1_list = pc_1['neighbors_idx']
        
        pool_idx_0_list = pc_0['pool_idx']
        pool_idx_1_list = pc_1['pool_idx']
        
        knearst_idx_0_to_1 = pc_0['knearst_idx_in_another_pc']
        knearst_idx_1_to_0 = pc_1['knearst_idx_in_another_pc']
        
        # Hierarchical feature extraction
        features_0_list = self.encoder_0(xyz_0_list, feat_0, neighbors_idx_0_list)
        features_1_list = self.encoder_1(xyz_1_list, feat_1, neighbors_idx_1_list)
        
        # Point set difference modeling at each level
        diff_features_0_list = []
        diff_features_1_list = []
        
        for i, (psdm, feat_0, feat_1) in enumerate(zip(self.psdm_modules, features_0_list, features_1_list)):
            # Use the cross-KNN indices at the corresponding level
            diff_0, diff_1 = psdm(feat_0, feat_1, knearst_idx_0_to_1, knearst_idx_1_to_0)
            diff_features_0_list.append(diff_0)
            diff_features_1_list.append(diff_1)
        
        # Combine original features with difference features
        combined_features_0_list = [torch.cat([f, d], dim=-1) for f, d in zip(features_0_list, diff_features_0_list)]
        combined_features_1_list = [torch.cat([f, d], dim=-1) for f, d in zip(features_1_list, diff_features_1_list)]
        
        # Update features_list with combined features
        features_0_list = combined_features_0_list
        features_1_list = combined_features_1_list
        
        # Feature decoding
        decoded_features_0 = self.decoder_0(features_0_list, pool_idx_0_list)
        decoded_features_1 = self.decoder_1(features_1_list, pool_idx_1_list)
        
        # Combine features from both point clouds
        combined_features = torch.cat([decoded_features_0, decoded_features_1], dim=-1)
        
        # Apply change detection head
        batch_size, num_points, feature_dim = combined_features.shape
        combined_features = combined_features.reshape(batch_size * num_points, -1)
        
        # Apply change detection head
        logits = self.change_head(combined_features)
        logits = logits.reshape(batch_size, num_points, -1)
        
        return {'logits': logits}


# Factory function for creating model
def get_model(num_classes: int = 2, **kwargs) -> ThreeCDNet:
    """Factory function for creating 3DCDNet model.
    
    Args:
        num_classes: Number of output classes
        **kwargs: Additional arguments to pass to model constructor
        
    Returns:
        ThreeCDNet model instance
    """
    return ThreeCDNet(num_classes=num_classes, **kwargs) 