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
    ):
        super(SiameseKPConv, self).__init__()
        self._num_classes = out_channels
        
        # Building the encoder (down modules)
        self.down_modules = nn.ModuleList()
        current_channels = in_channels
        
        for channels in down_channels:
            self.down_modules.append(
                SimpleBlock(
                    down_conv_nn=[current_channels, channels],
                    point_influence=point_influence,
                    bn_momentum=bn_momentum,
                    add_one=False,
                )
            )
            current_channels = channels
        
        # Inner module as Identity (can be changed if needed)
        self.inner_modules = nn.ModuleList([nn.Identity()])
        
        # Building the decoder (up modules)
        self.up_modules = nn.ModuleList()
        reversed_channels = list(reversed(down_channels))
        
        for i in range(len(reversed_channels) - 1):
            self.up_modules.append(
                SimpleBlock(
                    down_conv_nn=[reversed_channels[i], reversed_channels[i+1]],
                    point_influence=point_influence,
                    bn_momentum=bn_momentum,
                    add_one=False,
                )
            )
        
        # Final MLP for classification - returning raw logits
        self.FC_layer = nn.Sequential(
            nn.Linear(up_channels[-1], 64, bias=False),
            FastBatchNorm1d(64, momentum=bn_momentum),
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
            nn_list = knn(pos1, pos2, 1, batch1, batch2)
            
            # Calculate difference features
            diff = x2.clone()
            diff = x2 - x1[nn_list[1, :], :]
            
            # Save difference features for skip connections
            stack_down.append(diff)
        
        # Process last down module
        x1 = self.down_modules[-1](x1, pos1, batch1, pos1, batch1, k)
        x2 = self.down_modules[-1](x2, pos2, batch2, pos2, batch2, k)
        
        # Get difference features
        nn_list = knn(pos1, pos2, 1, batch1, batch2)
        x = x2.clone()
        x = x2 - x1[nn_list[1, :], :]
        
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
