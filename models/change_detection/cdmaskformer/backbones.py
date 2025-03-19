"""
Backbone networks for CDMaskFormer.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Optional, Tuple

def get_resnet(name: str, pretrained: bool = False) -> nn.Module:
    """
    Get a ResNet backbone with specified name.
    
    Args:
        name: Name of the ResNet backbone, e.g. 'resnet18', 'resnet50'.
        pretrained: Whether to use pretrained weights.
        
    Returns:
        The ResNet backbone.
    """
    if name == 'resnet18':
        return models.resnet18(pretrained=pretrained)
    elif name == 'resnet50':
        return models.resnet50(pretrained=pretrained)
    else:
        raise ValueError(f"Backbone {name} not supported")

class ResNetBackbone(nn.Module):
    """
    ResNet backbone for change detection.
    
    Extracts multi-scale features from a ResNet backbone.
    """
    def __init__(self, name: str, pretrained: bool = False):
        """
        Initialize ResNet backbone.
        
        Args:
            name: Name of the ResNet backbone, e.g. 'resnet18', 'resnet50'.
            pretrained: Whether to use pretrained weights.
        """
        super().__init__()
        self.resnet = get_resnet(name, pretrained)
        
        # Define output channels for different stages
        if name == 'resnet18':
            self.out_channels = {
                'res2': 64,     # layer1
                'res3': 128,    # layer2
                'res4': 256,    # layer3
                'res5': 512     # layer4
            }
        elif name == 'resnet50':
            self.out_channels = {
                'res2': 256,    # layer1
                'res3': 512,    # layer2
                'res4': 1024,   # layer3
                'res5': 2048    # layer4
            }
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], 
                                                                 Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the backbone.
        
        Args:
            x1: First input tensor of shape (B, C, H, W).
            x2: Second input tensor of shape (B, C, H, W).
            
        Returns:
            Tuple of (features1, features2), where each features is a tuple of (res2, res3, res4, res5).
        """
        def extract_features(x):
            # Initial layers
            x = self.resnet.conv1(x)
            x = self.resnet.bn1(x)
            x = self.resnet.relu(x)
            x = self.resnet.maxpool(x)
            
            # ResNet blocks
            res2 = self.resnet.layer1(x)
            res3 = self.resnet.layer2(res2)
            res4 = self.resnet.layer3(res3)
            res5 = self.resnet.layer4(res4)
            
            return (res2, res3, res4, res5)
        
        # Extract features from both images
        features1 = extract_features(x1)
        features2 = extract_features(x2)
        
        return (features1, features2)
    
    def get_channels(self) -> List[int]:
        """
        Get the number of channels for each output feature.
        
        Returns:
            List of channel numbers for each feature level.
        """
        return [
            self.out_channels['res2'],
            self.out_channels['res3'],
            self.out_channels['res4'],
            self.out_channels['res5']
        ]
