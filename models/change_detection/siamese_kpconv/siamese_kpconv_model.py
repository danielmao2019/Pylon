"""
SiameseKPConv model for 3D point cloud change detection

This module provides the implementation of SiameseKPConv, a siamese network 
based on KPConv for 3D point cloud change detection.
"""
from typing import Dict
import torch
import torch.nn as nn

from models.change_detection.siamese_kpconv.convolution_ops import SimpleBlock, FastBatchNorm1d
from models.change_detection.siamese_kpconv.utils import knn


class SiameseKPConv(nn.Module):
    """
    Siamese KPConv network for change detection in point clouds.
    This is a standalone implementation without torch_points3d dependencies.
    
    Args:
        in_channels: Number of input features
        out_channels: Number of output classes
        point_influence: Influence distance of points
        down_channels: List of channel dimensions for the encoder
        up_channels: List of channel dimensions for the decoder
        bn_momentum: Momentum for batch normalization
        dropout: Dropout probability for the final classifier
        inner_modules: List of inner modules to use between encoder and decoder.
                       If None, defaults to nn.Identity.
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
    ):
        super(SiameseKPConv, self).__init__()
        self._num_classes = out_channels
        self.point_influence = point_influence
        self.bn_momentum = bn_momentum
        
        # Initialize the encoder, inner modules, and decoder
        self._init_down_modules(in_channels, down_channels)
        self._init_inner_modules(inner_modules)
        self._init_up_modules(up_channels)
        self._init_final_mlp(up_channels[-1], out_channels, dropout)
    
    def _create_block(self, in_channels, out_channels):
        """Helper method to create a SimpleBlock with consistent parameters"""
        return SimpleBlock(
            down_conv_nn=[in_channels, out_channels],
            point_influence=self.point_influence,
            bn_momentum=self.bn_momentum,
            add_one=False,
        )
    
    def _init_down_modules(self, in_channels, down_channels):
        """Initialize the encoder (down modules)"""
        self.down_modules = nn.ModuleList()
        current_channels = in_channels
        
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
        
        for i in range(len(up_channels) - 1):
            self.up_modules.append(self._create_block(up_channels[i], up_channels[i+1]))
    
    def _init_final_mlp(self, in_channels, out_channels, dropout):
        """Initialize the final MLP for classification"""
        self.FC_layer = nn.Sequential(
            nn.Linear(in_channels, 64, bias=False),
            FastBatchNorm1d(64, momentum=self.bn_momentum),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=dropout) if dropout else nn.Identity(),
            nn.Linear(64, out_channels, bias=False)
        )
    
    def forward(self, inputs: Dict[str, torch.Tensor], k: int = 16) -> torch.Tensor:
        """
        Forward pass of the SiameseKPConv network
        
        Args:
            inputs: Dictionary containing:
                - 'pc_0': Dictionary with keys:
                    - 'pos': Point positions tensor [N, 3]
                    - 'x': Point features tensor [N, C]
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
            for subkey in ['pos', 'x', 'batch']:
                assert subkey in pc, f"Point cloud {key} missing required key: '{subkey}'"
                assert isinstance(pc[subkey], torch.Tensor), f"Expected {key}[{subkey}] to be a tensor, got {type(pc[subkey])}"
            
            # Check dimensions
            assert pc['pos'].dim() == 2 and pc['pos'].size(1) == 3, f"Expected {key}['pos'] to have shape [N, 3], got {pc['pos'].shape}"
            assert pc['x'].dim() == 2, f"Expected {key}['x'] to have shape [N, C], got {pc['x'].shape}"
            assert pc['batch'].dim() == 1, f"Expected {key}['batch'] to have shape [N], got {pc['batch'].shape}"
            assert pc['pos'].size(0) == pc['x'].size(0) == pc['batch'].size(0), f"Inconsistent sizes in {key}: pos={pc['pos'].size(0)}, x={pc['x'].size(0)}, batch={pc['batch'].size(0)}"
        
        # Check feature dimensions
        assert inputs['pc_0']['x'].size(1) == self.down_modules[0].kp_conv.num_inputs - (1 if self.down_modules[0].kp_conv.add_one else 0), \
            f"Expected input features to have {self.down_modules[0].kp_conv.num_inputs} channels, got {inputs['pc_0']['x'].size(1)}"
        
        # Prepare data
        data1 = inputs['pc_0']
        data2 = inputs['pc_1']
        
        # Process features
        pos1, x1, batch1 = data1['pos'], data1['x'], data1['batch']
        pos2, x2, batch2 = data2['pos'], data2['x'], data2['batch']
        
        # Stack for tracking down features
        stack_down = []
        
        # Encoder (processing both point clouds)
        for i in range(len(self.down_modules) - 1):
            # Process point cloud 1
            x1 = self.down_modules[i](x1, pos1, batch1, pos1, batch1, k)
            
            # Process point cloud 2
            x2 = self.down_modules[i](x2, pos2, batch2, pos2, batch2, k)
            
            # Find nearest neighbors from cloud1 to cloud2
            row_idx, col_idx = knn(pos1, pos2, 1, batch1, batch2)
            
            # Calculate difference features
            diff = x2.clone()
            diff = x2 - x1[row_idx]
            
            # Save difference features for skip connections
            stack_down.append(diff)
        
        # Process last down module
        x1 = self.down_modules[-1](x1, pos1, batch1, pos1, batch1, k)
        x2 = self.down_modules[-1](x2, pos2, batch2, pos2, batch2, k)
        
        # Get difference features
        row_idx, col_idx = knn(pos1, pos2, 1, batch1, batch2)
        x = x2.clone()
        x = x2 - x1[row_idx]
        
        # Inner module (identity by default)
        if not isinstance(self.inner_modules[0], nn.Identity):
            stack_down.append(x)
            x = self.inner_modules[0](x)
            innermost = True
        else:
            innermost = False
        
        # Decoder with skip connections
        for i in range(len(self.up_modules)):
            if i == 0 and innermost:
                x = torch.cat([x, stack_down.pop()], dim=1)
                x = self.up_modules[i](x, pos2, batch2, pos2, batch2, k)
            else:
                x = torch.cat([x, stack_down.pop()], dim=1)
                x = self.up_modules[i](x, pos2, batch2, pos2, batch2, k)
        
        # Final classification - returning raw logits
        self.last_feature = x
        output = self.FC_layer(self.last_feature)
        
        return output
