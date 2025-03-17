from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .blocks import BasicConv2d, ChangeGuideModule


class HCGMNet(nn.Module):
    """
    HCGMNet: A Hierarchical Change Guiding Map Network for Change Detection
    
    This model uses a VGG16 backbone with a hierarchical structure and a Change Guide Module (CGM)
    for attention-based change detection.
    
    Reference:
        HCGMNET: A HIERARCHICAL CHANGE GUIDING MAP NETWORK FOR CHANGE DETECTION,
        IGARSS 2023, Oral. Chengxi. Han, Chen WU, Do Du
        https://arxiv.org/abs/2302.10420
    
    Args:
        num_classes (int, optional): Number of output classes. Default: 2.
    """
    def __init__(self, num_classes: int = 2):
        super(HCGMNet, self).__init__()
        # Initialize VGG16 backbone with pretrained weights
        vgg16_bn = models.vgg16_bn(pretrained=True)
        self.inc = vgg16_bn.features[:5]  # 64
        self.down1 = vgg16_bn.features[5:12]  # 128
        self.down2 = vgg16_bn.features[12:22]  # 256
        self.down3 = vgg16_bn.features[22:32]  # 512
        self.down4 = vgg16_bn.features[32:42]  # 512

        # Feature reduction layers
        self.conv_reduce_1 = BasicConv2d(128*2, 128, 3, 1, 1)
        self.conv_reduce_2 = BasicConv2d(256*2, 256, 3, 1, 1)
        self.conv_reduce_3 = BasicConv2d(512*2, 512, 3, 1, 1)
        self.conv_reduce_4 = BasicConv2d(512*2, 512, 3, 1, 1)

        # Upsampling layers
        self.up_layer4 = BasicConv2d(512, 512, 3, 1, 1)
        self.up_layer3 = BasicConv2d(512, 512, 3, 1, 1)
        self.up_layer2 = BasicConv2d(256, 256, 3, 1, 1)

        # Decoder for intermediate change map
        self.deocde = nn.Sequential(
            BasicConv2d(1408, 512, 3, 1, 1),
            BasicConv2d(512, 256, 3, 1, 1),
            BasicConv2d(256, 64, 3, 1, 1),
            nn.Conv2d(64, num_classes, 3, 1, 1)
        )

        # Decoder for final change map
        self.deocde_final = nn.Sequential(
            BasicConv2d(1408, 512, 3, 1, 1),
            BasicConv2d(512, 256, 3, 1, 1),
            BasicConv2d(256, 64, 3, 1, 1),
            nn.Conv2d(64, num_classes, 3, 1, 1)
        )

        # Change Guide Modules
        self.cgm_2 = ChangeGuideModule(256)
        self.cgm_3 = ChangeGuideModule(512)
        self.cgm_4 = ChangeGuideModule(512)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the HCGMNet model.
        
        Args:
            inputs (Dict[str, torch.Tensor]): Dictionary containing input tensors.
                Expected keys:
                - 'img_1': First image tensor of shape [B, C, H, W]
                - 'img_2': Second image tensor of shape [B, C, H, W]
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing output tensors.
                During training:
                - 'change_map': Final change map logits of shape [B, num_classes, H, W]
                - 'aux_change_map': Intermediate change map logits of shape [B, num_classes, H, W]
                During inference:
                - 'change_map': Final change map logits of shape [B, num_classes, H, W]
        """
        # Extract inputs
        A = inputs['img_1']
        B = inputs['img_2']
        
        # Get original size for later upsampling
        size = A.size()[2:]
        
        # Extract features from first image
        layer1_pre = self.inc(A)
        layer1_A = self.down1(layer1_pre)
        layer2_A = self.down2(layer1_A)
        layer3_A = self.down3(layer2_A)
        layer4_A = self.down4(layer3_A)

        # Extract features from second image
        layer1_pre = self.inc(B)
        layer1_B = self.down1(layer1_pre)
        layer2_B = self.down2(layer1_B)
        layer3_B = self.down3(layer2_B)
        layer4_B = self.down4(layer3_B)

        # Concatenate features from both images
        layer1 = torch.cat((layer1_B, layer1_A), dim=1)
        layer2 = torch.cat((layer2_B, layer2_A), dim=1)
        layer3 = torch.cat((layer3_B, layer3_A), dim=1)
        layer4 = torch.cat((layer4_B, layer4_A), dim=1)

        # Reduce feature dimensions
        layer1 = self.conv_reduce_1(layer1)
        layer2 = self.conv_reduce_2(layer2)
        layer3 = self.conv_reduce_3(layer3)
        layer4 = self.conv_reduce_4(layer4)

        # Apply upsampling layers
        layer4 = self.up_layer4(layer4)
        layer3 = self.up_layer3(layer3)
        layer2 = self.up_layer2(layer2)

        # Upsample features to match layer1 size
        layer4_1 = F.interpolate(layer4, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer3_1 = F.interpolate(layer3, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer2_1 = F.interpolate(layer2, layer1.size()[2:], mode='bilinear', align_corners=True)

        # Fuse features and generate intermediate change map
        feature_fuse = torch.cat((layer1, layer2_1, layer3_1, layer4_1), dim=1)
        change_map = self.deocde(feature_fuse)

        # Apply Change Guide Modules
        layer2 = self.cgm_2(layer2, change_map)
        layer3 = self.cgm_3(layer3, change_map)
        layer4 = self.cgm_4(layer4, change_map)

        # Upsample guided features
        layer4_2 = F.interpolate(layer4, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer3_2 = F.interpolate(layer3, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer2_2 = F.interpolate(layer2, layer1.size()[2:], mode='bilinear', align_corners=True)

        # Fuse guided features and generate final change map
        new_feature_fuse = torch.cat((layer1, layer2_2, layer3_2, layer4_2), dim=1)
        
        # Upsample change maps to original input size
        change_map = F.interpolate(change_map, size, mode='bilinear', align_corners=True)
        final_map = self.deocde_final(new_feature_fuse)
        final_map = F.interpolate(final_map, size, mode='bilinear', align_corners=True)

        # Return different outputs based on training mode
        if self.training:
            return {
                'change_map': final_map,
                'aux_change_map': change_map
            }
        return {
            'change_map': final_map
        }
