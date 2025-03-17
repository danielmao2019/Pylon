from typing import Dict, Union
import torch
import torch.nn as nn
from models.change_detection.cdmaskformer.backbones import ResNetBackbone
from models.change_detection.cdmaskformer.models.cdmask import CDMask


class CDMaskFormer(nn.Module):
    def __init__(self, backbone_name, backbone_args, cdmask_args):
        super().__init__()
        
        # Create backbone
        self.backbone = ResNetBackbone(
            name=backbone_name,
            pretrained=backbone_args.get('pretrained', False)
        )
        
        # Get backbone channels
        channels = self.backbone.get_channels()
        
        # Update cdmask_args with backbone channels
        cdmask_args['channels'] = channels
        
        self.cdmask = CDMask(**cdmask_args)
        
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        # Extract features from backbone
        img1 = inputs['img_1']
        img2 = inputs['img_2']
        
        # Get features from both images
        img1_features = self.backbone(img1)
        img2_features = self.backbone(img2)
        
        # Format features for CDMask
        features = {
            'featuresA': [
                img1_features['res2'],
                img1_features['res3'],
                img1_features['res4'],
                img1_features['res5']
            ],
            'featuresB': [
                img2_features['res2'],
                img2_features['res3'],
                img2_features['res4'],
                img2_features['res5']
            ]
        }
        
        # Forward through CDMask - output format depends on training mode
        return self.cdmask(features) 