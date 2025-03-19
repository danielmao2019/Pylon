"""
Backbone networks for CDMaskFormer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import torch.utils.model_zoo as model_zoo
import os

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnet18stem': 'https://download.openmmlab.com/pretrain/third_party/resnet18_v1c-b5776b93.pth',
    'resnet50stem': 'https://download.openmmlab.com/pretrain/third_party/resnet50_v1c-2cccc1ad.pth',
    'resnet101stem': 'https://download.openmmlab.com/pretrain/third_party/resnet101_v1c-e67eebb6.pth',
}

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, dilation: int = 1, downsample: Optional[nn.Module] = None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, dilation: int = 1, downsample: Optional[nn.Module] = None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu_inplace(out)

        return out

class Resnet(nn.Module):
    def __init__(self, block: nn.Module, layers: List[int], out_stride: int = 8, use_stem: bool = False, stem_channels: int = 64, in_channels: int = 3):
        super(Resnet, self).__init__()
        self.inplanes = 64
        self.out_stride = out_stride

        # Configurable output stride (8, 16, or 32)
        outstride_to_strides_and_dilations = {
            8: ((1, 2, 1, 1), (1, 1, 2, 4)),
            16: ((1, 2, 2, 1), (1, 1, 1, 2)),
            32: ((1, 2, 2, 2), (1, 1, 1, 1)),
        }
        stride_list, dilation_list = outstride_to_strides_and_dilations[out_stride]

        # Optional stem block
        if use_stem:
            self.stem = nn.Sequential(
                conv3x3(in_channels, stem_channels//2, stride=2),
                nn.BatchNorm2d(stem_channels//2),
                nn.ReLU(inplace=False),
                conv3x3(stem_channels//2, stem_channels//2),
                nn.BatchNorm2d(stem_channels//2),
                nn.ReLU(inplace=False),
                conv3x3(stem_channels//2, stem_channels),
                nn.BatchNorm2d(stem_channels),
                nn.ReLU(inplace=False)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, stem_channels, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(stem_channels)
            self.relu = nn.ReLU(inplace=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks=layers[0], stride=stride_list[0], dilation=dilation_list[0])
        self.layer2 = self._make_layer(block, 128, blocks=layers[1], stride=stride_list[1], dilation=dilation_list[1])
        self.layer3 = self._make_layer(block, 256, blocks=layers[2], stride=stride_list[2], dilation=dilation_list[2])
        self.layer4 = self._make_layer(block, 512, blocks=layers[3], stride=stride_list[3], dilation=dilation_list[3])

    def _make_layer(self, block: nn.Module, planes: int, blocks: int, stride: int = 1, dilation: int = 1, contract_dilation=True) -> nn.Sequential:
        downsample = None
        dilations = [dilation] * blocks

        if contract_dilation and dilation > 1: dilations[0] = dilation // 2
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilations[0], downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilations[i]))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if hasattr(self, 'stem'):
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        
        x = self.maxpool(x)
        
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        outs = [x1, x2, x3, x4]

        return tuple(outs)

def get_resnet(name: str, pretrained: bool = False) -> Resnet:
    """Get a ResNet model with specified configuration."""
    resnet_configs = {
        'resnet18': (BasicBlock, [2, 2, 2, 2]),
        'resnet34': (BasicBlock, [3, 4, 6, 3]),
        'resnet50': (Bottleneck, [3, 4, 6, 3]),
        'resnet101': (Bottleneck, [3, 4, 23, 3]),
        'resnet152': (Bottleneck, [3, 8, 36, 3])
    }
    
    if name not in resnet_configs:
        raise ValueError(f"ResNet variant '{name}' not supported. Supported variants: {list(resnet_configs.keys())}")
    
    block, layers = resnet_configs[name]
    model = Resnet(block, layers)
    
    if pretrained:
        # Load pretrained weights from torchvision
        import torchvision.models as models
        pretrained_model = getattr(models, name)(pretrained=True)
        model.load_state_dict(pretrained_model.state_dict())
    
    return model

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
        if name == 'resnet18':
            self.resnet = get_resnet(name, pretrained)
        elif name == 'resnet50_OS8':
            self.resnet = get_resnet50_OS8(pretrained)
        elif name == 'resnet50_OS32':
            self.resnet = get_resnet50_OS32(pretrained)
        else:
            raise ValueError(f"Backbone {name} not supported")
        
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
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the backbone.
        
        Args:
            x1: First input tensor of shape (B, C, H, W).
            x2: Second input tensor of shape (B, C, H, W).
            
        Returns:
            Tuple of (res2, res3, res4, res5) features for both images
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
        
        return features1, features2
    
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

def get_resnet18(pretrained=True):
    model = Resnet(BasicBlock, [2, 2, 2, 2], out_stride=32)
    if pretrained:
        checkpoint = model_zoo.load_url(model_urls['resnet18'])
        if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
        else: state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    return model

def get_resnet50_OS8(pretrained=True):
    model = Resnet(Bottleneck, [3, 4, 6, 3], out_stride=8, use_stem=True)
    if pretrained:
        checkpoint = model_zoo.load_url(model_urls['resnet50stem'])
        if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
        else: state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    return model

def get_resnet50_OS32(pretrained=True):
    model = Resnet(Bottleneck, [3, 4, 6, 3], out_stride=32, use_stem=False)
    if pretrained:
        checkpoint = model_zoo.load_url(model_urls['resnet50'])
        if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
        else: state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    return model
