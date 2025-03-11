"""
SiameseKPConv model for 3D point cloud change detection

This module provides the implementation of SiameseKPConv, a siamese network 
based on KPConv for 3D point cloud change detection. The model takes two point clouds
representing the same area at different times and detects changes between them.

The implementation is based on the original torch-points3d-SiameseKPConv repository
(https://github.com/humanpose1/torch-points3d-SiameseKPConv), but adapted to work 
as a standalone module without dependencies on the torch_points3d framework.

Architecture:
- Encoder: Processes both point clouds with shared weights through KPConv blocks
- Feature differencing: Computes differences between corresponding points using kNN
- Decoder: Processes the difference features with skip connections from the encoder
- Final MLP: Classifies each point as changed or unchanged

The model supports various KPConv block types:
- SimpleBlock: Basic KPConv block with convolution -> BN -> activation
- ResnetBBlock: Bottleneck Resnet block for KPConv with residual connections
- KPDualBlock: Combination of blocks for more complex processing

Reference paper:
"SiameseKPConv: A Siamese KPConv Network Architecture for 3D Point Cloud Change Detection"
"""
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn

from models.change_detection.siamese_kpconv.convolution_ops import (
    SimpleBlock, ResnetBBlock, KPDualBlock, FastBatchNorm1d
)
from models.change_detection.siamese_kpconv.utils import knn, gather, add_ones


class SiameseKPConv(nn.Module):
    """
    Siamese KPConv network for change detection in point clouds.
    
    This model processes two point clouds representing the same area at different times
    and detects changes between them. It uses a siamese architecture where both point clouds
    are processed by the same encoder, then features are differenced and processed by a decoder
    to produce change detection outputs.
    
    The architecture follows a U-Net style with skip connections, and uses the KPConv 
    (Kernel Point Convolution) operator for point cloud processing, which defines 
    convolution kernels as a set of kernel points in space.
    
    Args:
        in_channels: Number of input features per point (e.g., 3 for RGB)
        out_channels: Number of output classes (typically 2 for binary change detection)
        point_influence: Radius of influence around each kernel point
        down_channels: List of feature dimensions for each layer in the encoder
        up_channels: List of feature dimensions for each layer in the decoder
        bn_momentum: Momentum parameter for batch normalization
        dropout: Dropout probability in the final classification layer
        inner_modules: Optional modules to use between encoder and decoder
                       If None, defaults to nn.Identity
        conv_type: Type of convolution blocks to use:
                  - 'simple': Basic KPConv block with convolution -> BN -> activation
                  - 'resnet': Bottleneck Resnet block for KPConv with residual connections
                  - 'dual': Combination of blocks for more complex processing
        block_params: Additional parameters for blocks when using 'dual' conv_type
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        point_influence: float = 0.025,
        down_channels: list = [32, 64, 128, 256],
        up_channels: list = [256, 128, 64, 32],
        bn_momentum: float = 0.02,
        dropout: float = 0.1,
        inner_modules: list = None,
        conv_type: str = "simple",
        block_params: dict = None,
    ):
        super(SiameseKPConv, self).__init__()
        self._num_classes = out_channels
        self.point_influence = point_influence
        self.bn_momentum = bn_momentum
        self.conv_type = conv_type.lower()
        self.block_params = block_params or {}
        self.in_channels = in_channels
        
        # Initialize the encoder, inner modules, and decoder
        self._init_down_modules(in_channels, down_channels)
        self._init_inner_modules(inner_modules)
        self._init_up_modules(up_channels)
        self._init_final_mlp(up_channels[-1], out_channels, dropout)
        
        # Store last feature for potential use
        self.last_feature = None
    
    def _create_block(self, in_channels, out_channels):
        """Helper method to create a convolution block based on the specified type"""
        if self.conv_type == "simple":
            return SimpleBlock(
                down_conv_nn=[in_channels, out_channels],
                point_influence=self.point_influence,
                bn_momentum=self.bn_momentum,
            )
        elif self.conv_type == "resnet":
            return ResnetBBlock(
                down_conv_nn=[in_channels, out_channels],
                point_influence=self.point_influence,
                bn_momentum=self.bn_momentum,
                has_bottleneck=True,
            )
        elif self.conv_type == "dual":
            # Extract block parameters for dual blocks
            block_names = self.block_params.get("block_names", ["SimpleBlock", "ResnetBBlock"])
            has_bottleneck = self.block_params.get("has_bottleneck", [False, True])
            max_num_neighbors = self.block_params.get("max_num_neighbors", [16, 16])
            
            return KPDualBlock(
                block_names=block_names,
                down_conv_nn=[[in_channels, out_channels], [out_channels, out_channels]],
                point_influence=self.point_influence,
                has_bottleneck=has_bottleneck,
                max_num_neighbors=max_num_neighbors,
                bn_momentum=self.bn_momentum,
            )
        else:
            raise ValueError(f"Unknown conv_type: {self.conv_type}. Choose from 'simple', 'resnet', or 'dual'.")
    
    def _init_down_modules(self, in_channels, down_channels):
        """Initialize the encoder (down modules)"""
        self.down_modules = nn.ModuleList()
        current_channels = in_channels
        
        # Create each down module
        for channels in down_channels:
            self.down_modules.append(self._create_block(current_channels, channels))
            current_channels = channels
    
    def _init_inner_modules(self, inner_modules):
        """Initialize the inner modules"""
        if inner_modules is None:
            # Default to Identity if no inner modules are specified
            self.inner_modules = nn.ModuleList([nn.Identity()])
        else:
            # Use the provided inner modules
            self.inner_modules = nn.ModuleList(inner_modules)
    
    def _init_up_modules(self, up_channels):
        """Initialize the decoder (up modules)"""
        self.up_modules = nn.ModuleList()
        
        # Create each up module
        for i in range(len(up_channels) - 1):
            # For each up module, the input dimension is the current up_channel
            # plus the corresponding skip connection from the encoder
            # The output dimension is the next up_channel
            in_channels = up_channels[i]
            out_channels = up_channels[i+1]
            self.up_modules.append(self._create_block(in_channels, out_channels))
    
    def _init_final_mlp(self, in_channels, out_channels, dropout):
        """Initialize the final MLP for classification"""
        self.FC_layer = nn.Sequential(
            nn.Linear(in_channels, 64, bias=False),
            FastBatchNorm1d(64, momentum=self.bn_momentum),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=dropout) if dropout else nn.Identity(),
            nn.Linear(64, out_channels, bias=False)
            # No activation - returning raw logits as required
        )
    
    def forward(self, inputs: Dict[str, torch.Tensor], k: int = 16) -> torch.Tensor:
        """
        Forward pass of the SiameseKPConv network
        
        Args:
            inputs: Dictionary containing:
                - 'pc_0': Dictionary with keys:
                    - 'pos': Point positions tensor [N, 3]
                    - 'feat': Point features tensor [N, C]
                    - 'batch': Batch indices [N]
                - 'pc_1': Dictionary with the same structure as pc_0
            k: Number of neighbors to use in kNN
            
        Returns:
            Change detection logits [N, num_classes]
        """
        # Input validation
        required_keys = ['pc_0', 'pc_1']
        for key in required_keys:
            assert key in inputs, f"Input missing required key: '{key}'"
            pc = inputs[key]
            assert isinstance(pc, dict), f"Expected {key} to be a dictionary, got {type(pc)}"
            for subkey in ['pos', 'feat', 'batch']:
                assert subkey in pc, f"Point cloud {key} missing required key: '{subkey}'"
                assert isinstance(pc[subkey], torch.Tensor), f"Expected {key}[{subkey}] to be a tensor, got {type(pc[subkey])}"
            
            # Check dimensions
            assert pc['pos'].dim() == 2 and pc['pos'].size(1) == 3, f"Expected {key}['pos'] to have shape [N, 3], got {pc['pos'].shape}"
            assert pc['feat'].dim() == 2, f"Expected {key}['feat'] to have shape [N, C], got {pc['feat'].shape}"
            assert pc['batch'].dim() == 1, f"Expected {key}['batch'] to have shape [N], got {pc['batch'].shape}"
            assert pc['pos'].size(0) == pc['feat'].size(0) == pc['batch'].size(0), f"Inconsistent sizes in {key}: pos={pc['pos'].size(0)}, feat={pc['feat'].size(0)}, batch={pc['batch'].size(0)}"
        
        # Check feature dimensions
        assert inputs['pc_0']['feat'].size(1) == 1, f"Expected input features to have 1 channel (ones), got {inputs['pc_0']['feat'].size(1)}"
        
        # Prepare data
        data1 = inputs['pc_0']
        data2 = inputs['pc_1']
        
        # Process features - concatenate position and features
        pos1, feat1, batch1 = data1['pos'], data1['feat'], data1['batch']
        pos2, feat2, batch2 = data2['pos'], data2['feat'], data2['batch']
        
        # Concatenate position and features for KPConv processing
        x1 = torch.cat([pos1, feat1], dim=1)  # [N, 4] (xyz + ones)
        x2 = torch.cat([pos2, feat2], dim=1)  # [N, 4] (xyz + ones)
        
        # Stack for tracking down features
        stack_down = []
        
        # Encoder (processing both point clouds)
        for i in range(len(self.down_modules) - 1):
            # Process point cloud 1
            x1 = self.down_modules[i](x1, pos1, batch1, pos1, batch1, k)
            
            # Process point cloud 2
            x2 = self.down_modules[i](x2, pos2, batch2, pos2, batch2, k)
            
            # Find nearest neighbors from cloud2 to cloud1 (since we want to predict changes in cloud2)
            row_idx, col_idx = knn(pos2, pos1, 1, batch2, batch1)
            
            # Calculate difference features
            diff = x2.clone()
            diff = x2 - x1[col_idx]
            
            # Save difference features for skip connections
            stack_down.append(diff)
        
        # Process last down module
        x1 = self.down_modules[-1](x1, pos1, batch1, pos1, batch1, k)
        x2 = self.down_modules[-1](x2, pos2, batch2, pos2, batch2, k)
        
        # Get difference features
        row_idx, col_idx = knn(pos2, pos1, 1, batch2, batch1)
        x = x2.clone()
        x = x2 - x1[col_idx]
        
        # Inner module (identity by default)
        if not isinstance(self.inner_modules[0], nn.Identity):
            stack_down.append(x)
            x = self.inner_modules[0](x)
        
        # Process up modules with proper skip connections
        for i, up_module in enumerate(self.up_modules):
            # Get corresponding skip connection from encoder
            skip_idx = len(stack_down) - i - 1
            if skip_idx >= 0:
                # Concatenate skip connection features
                skip_x = stack_down[skip_idx]
                x = torch.cat([x, skip_x], dim=1)
            
            # Apply up module
            x = up_module(x, pos2, batch2, pos2, batch2, k)
        
        # Store the last feature for potential external use
        self.last_feature = x
        
        # Final classification
        output = self.FC_layer(self.last_feature)
        
        return output
