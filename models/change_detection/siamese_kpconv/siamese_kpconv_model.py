"""
SiameseKPConv model for 3D point cloud change detection

This module provides the implementation of SiameseKPConv, a siamese network 
based on KPConv for 3D point cloud change detection.
"""
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn
from torch_geometric.data import Data

from models.change_detection.siamese_kpconv.convolution_ops import SimpleBlock, FastBatchNorm1d


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
        
        # Final MLP for classification
        self.FC_layer = nn.Sequential(
            nn.Linear(up_channels[-1], 64, bias=False),
            FastBatchNorm1d(64, momentum=bn_momentum),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=dropout) if dropout else nn.Identity(),
            nn.Linear(64, out_channels, bias=False),
            nn.LogSoftmax(dim=-1)
        )
        
    def forward(self, inputs: Dict[str, torch.Tensor], k: int = 16) -> torch.Tensor:
        """
        Forward pass of the SiameseKPConv network
        
        Args:
            inputs: Dictionary containing:
                - 'x1': First point cloud data object
                - 'x2': Second point cloud data object
            k: Number of neighbors to use in kNN
            
        Returns:
            Change detection predictions [N, num_classes]
        """
        # Prepare data
        data1 = inputs['x1']
        data2 = inputs['x2']
        
        # Process features
        pos1, x1, batch1 = data1.pos, data1.x, data1.batch
        pos2, x2, batch2 = data2.pos, data2.x, data2.batch
        
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
        
        # Final classification
        self.last_feature = x
        output = self.FC_layer(self.last_feature)
        
        return output
    
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, weight_classes=None) -> torch.Tensor:
        """
        Compute the loss for training
        
        Args:
            pred: Model predictions [N, num_classes]
            target: Ground truth labels [N]
            weight_classes: Class weights [num_classes]
            
        Returns:
            Loss value
        """
        if weight_classes is not None:
            return F.nll_loss(pred, target, weight=weight_classes)
        else:
            return F.nll_loss(pred, target)


# Example of usage
if __name__ == "__main__":
    # Create sample data
    pos1 = torch.rand(100, 3)  # 100 points with 3D coordinates
    x1 = torch.rand(100, 3)    # Features for each point (e.g., RGB color)
    batch1 = torch.zeros(100, dtype=torch.long)  # All points belong to the same batch
    
    pos2 = torch.rand(100, 3)  # 100 points with 3D coordinates
    x2 = torch.rand(100, 3)    # Features for each point
    batch2 = torch.zeros(100, dtype=torch.long)  # All points belong to the same batch
    
    # Create Data objects
    data1 = Data(pos=pos1, x=x1, batch=batch1)
    data2 = Data(pos=pos2, x=x2, batch=batch2)
    
    # Initialize model
    model = SiameseKPConv(
        in_channels=3,        # input feature dimension
        out_channels=2,       # number of classes (e.g., change/no change)
        point_influence=0.1   # influence distance of points
    )
    
    # Forward pass
    inputs = {'x1': data1, 'x2': data2}
    output = model(inputs, k=8)
    
    print(f"Model output shape: {output.shape}")  # Should be [100, 2] 