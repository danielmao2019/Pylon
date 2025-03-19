from typing import Dict
import torch
from torch import nn
from utils.builders import build_from_config


class CDMaskFormer(nn.Module):
    def __init__(self, backbone_cfg, head_cfg):
        super(CDMaskFormer, self).__init__()
        self.backbone = build_from_config(backbone_cfg)
        self.head = build_from_config(head_cfg)
    
    def forward(self, inputs: Dict[str, torch.Tensor], gtmask=None):
        x1, x2 = inputs['img_1'], inputs['img_2']
        backbone_outputs = self.backbone(x1, x2)
        if gtmask == None:
            x_list = self.head(backbone_outputs)
        else:
            x_list = self.head(backbone_outputs, gtmask)
        return x_list
