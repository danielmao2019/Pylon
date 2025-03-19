from typing import Dict, Union, List, Tuple
import torch
import torch.nn as nn
from models.change_detection.cdmaskformer.backbones import get_resnet18, get_resnet50_OS8, get_resnet50_OS32
from models.change_detection.cdmaskformer.models.cdmask import CDMask


class CDMaskFormer(nn.Module):
    def __init__(self, backbone_name, backbone_args, cdmask_args):
        super().__init__()
        
        # Create backbone
        if backbone_name == 'resnet18':
            self.backbone = get_resnet18(pretrained=backbone_args.get('pretrained', False))
            channels = [64, 128, 256, 512]
        elif backbone_name == 'resnet50_OS8':
            self.backbone = get_resnet50_OS8(pretrained=backbone_args.get('pretrained', False))
            channels = [256, 512, 1024, 2048]
        elif backbone_name == 'resnet50_OS32':
            self.backbone = get_resnet50_OS32(pretrained=backbone_args.get('pretrained', False))
            channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Backbone {backbone_name} not supported")
        
        # Update cdmask_args with backbone channels
        cdmask_args['channels'] = channels
        
        self.cdmask = CDMask(**cdmask_args)
        
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        # Extract features from backbone
        img1 = inputs['img_1']
        img2 = inputs['img_2']
        
        # Get features from both images at once
        features = self.backbone(img1, img2)
        
        # Forward through CDMask - output format depends on training mode
        output = self.cdmask(features)
        
        return output
