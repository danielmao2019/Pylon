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
from models.change_detection.siamese_kpconv.interpolate import KNNInterpolate


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
    
    For detailed documentation, see: docs/models/change_detection/siamese_kpconv.md
    
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
        conv_type: str = "dual",  # Changed default to match original
        block_params: dict = None,
    ):
        super(SiameseKPConv, self).__init__()
        self._num_classes = out_channels
        self.point_influence = point_influence
        self.bn_momentum = bn_momentum
        self.conv_type = conv_type.lower()
        
        # Default block parameters matching original implementation
        self.block_params = block_params or {
            "n_kernel_points": 25,
            "block_names": [
                ["SimpleBlock", "ResnetBBlock"],
                ["ResnetBBlock", "ResnetBBlock"],
                ["ResnetBBlock", "ResnetBBlock"],
                ["ResnetBBlock", "ResnetBBlock"],
            ],
            "has_bottleneck": [
                [False, True],
                [True, True],
                [True, True],
                [True, True],
            ],
            "max_num_neighbors": [
                [25, 25],
                [25, 30],
                [30, 38],
                [38, 38],
            ]
        }
        
        self.in_channels = in_channels
        self.down_channels = down_channels
        
        # Initialize the encoder, inner modules, and decoder
        self._init_down_modules(in_channels, down_channels)
        self._init_inner_modules(inner_modules)
        self._init_up_modules(up_channels)
        self._init_final_mlp(up_channels[-1], out_channels, dropout)
        
        # Initialize interpolation for feature propagation
        self.interpolate = KNNInterpolate(k=3)
        
        # Store last feature for potential use
        self.last_feature = None
    
    def _create_block(self, in_channels, out_channels, block_idx=None):
        """Helper method to create a convolution block based on the specified type"""
        if block_idx is not None and self.conv_type == "dual":
            # Use the exact block configuration from original implementation
            return KPDualBlock(
                block_names=self.block_params["block_names"][block_idx],
                down_conv_nn=[[in_channels, out_channels], [out_channels, out_channels]],
                point_influence=self.point_influence,
                has_bottleneck=self.block_params["has_bottleneck"][block_idx],
                max_num_neighbors=self.block_params["max_num_neighbors"][block_idx],
                bn_momentum=self.bn_momentum,
                n_kernel_points=self.block_params["n_kernel_points"],
            )
        elif self.conv_type == "simple":
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
        else:
            raise ValueError(f"Unknown conv_type: {self.conv_type}. Choose from 'simple', 'resnet', or 'dual'.")
    
    def _init_down_modules(self, in_channels, down_channels):
        """Initialize the encoder (down modules)"""
        self.down_modules = nn.ModuleList()
        
        # Create each down module with proper configuration
        for i in range(len(down_channels)):
            in_ch = in_channels if i == 0 else down_channels[i-1]
            out_ch = down_channels[i]
            self.down_modules.append(
                self._create_block(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    block_idx=i  # Keep block_idx for encoder as it uses specific configurations
                )
            )
    
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
        
        # Create each up module with proper feature dimensions
        for i in range(len(up_channels) - 1):
            in_ch = up_channels[i] + self.down_channels[-(i+2)]  # Include skip connection
            out_ch = up_channels[i + 1]
            self.up_modules.append(
                self._create_block(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    block_idx=None  # Decoder uses simpler block configuration
                )
            )
    
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
        
        # Prepare data
        data1 = inputs['pc_0']
        data2 = inputs['pc_1']
        
        # Process features - concatenate position and features
        pos1, feat1, batch1 = data1['pos'], data1['feat'], data1['batch']
        pos2, feat2, batch2 = data2['pos'], data2['feat'], data2['batch']
        
        # Concatenate position and features for KPConv processing
        x1 = torch.cat([pos1, feat1], dim=1)  # [N, 4] (xyz + ones)
        x2 = torch.cat([pos2, feat2], dim=1)  # [N, 4] (xyz + ones)
        
        # Store positions and features at each level for skip connections
        pos1_stack = [pos1]
        pos2_stack = [pos2]
        batch1_stack = [batch1]
        batch2_stack = [batch2]
        feat_stack = []  # Store difference features
        
        # Encoder (processing both point clouds)
        for i in range(len(self.down_modules) - 1):
            # Process point cloud 1
            x1 = self.down_modules[i](x1, pos1, batch1, pos1, batch1, k)
            
            # Process point cloud 2
            x2 = self.down_modules[i](x2, pos2, batch2, pos2, batch2, k)
            
            # Find nearest neighbors from cloud2 to cloud1
            row_idx, col_idx = knn(pos2, pos1, 1, batch2, batch1)
            
            # Calculate difference features
            diff = x2 - x1[col_idx]
            
            # Store features and positions for skip connections
            feat_stack.append(diff)
            pos1_stack.append(pos1)
            pos2_stack.append(pos2)
            batch1_stack.append(batch1)
            batch2_stack.append(batch2)
        
        # Process last down module
        x1 = self.down_modules[-1](x1, pos1, batch1, pos1, batch1, k)
        x2 = self.down_modules[-1](x2, pos2, batch2, pos2, batch2, k)
        
        # Get difference features for bottleneck
        row_idx, col_idx = knn(pos2, pos1, 1, batch2, batch1)
        x = x2 - x1[col_idx]
        
        # Inner module (identity by default)
        if not isinstance(self.inner_modules[0], nn.Identity):
            feat_stack.append(x)
            x = self.inner_modules[0](x)
        
        # Process up modules with proper feature interpolation and concatenation
        for i, up_module in enumerate(self.up_modules):
            # Get corresponding skip connection features and positions
            skip_feat = feat_stack[-(i+1)]
            skip_pos2 = pos2_stack[-(i+2)]  # Use positions from previous level
            skip_batch2 = batch2_stack[-(i+2)]
            
            # Interpolate current features to skip connection resolution
            x_interp = self.interpolate(
                skip_pos2, skip_batch2,  # query
                pos2, batch2,  # support
                x  # features to interpolate
            )
            
            # Concatenate with skip connection features
            x = torch.cat([x_interp, skip_feat], dim=1)
            
            # Update positions and batch indices for next layer
            pos2 = skip_pos2
            batch2 = skip_batch2
            
            # Apply up module
            x = up_module(x, pos2, batch2, pos2, batch2, k)
        
        # Store the last feature for potential external use
        self.last_feature = x
        
        # Final classification
        output = self.FC_layer(self.last_feature)
        
        return output
