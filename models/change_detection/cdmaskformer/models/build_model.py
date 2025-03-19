from typing import Dict
import torch
from torch import nn
from utils.builders import build_from_config


class CDMaskFormer(nn.Module):
    def __init__(self, cfg):
        super(CDMaskFormer, self).__init__()
        self.backbone = build_from_config(cfg.backbone)
        self.decoderhead = build_from_config(cfg.decoderhead)
    
    def forward(self, inputs: Dict[str, torch.Tensor], gtmask=None):
        x1, x2 = inputs['img_1'], inputs['img_2']
        backbone_outputs = self.backbone(x1, x2)
        if gtmask == None:
            x_list = self.decoderhead(backbone_outputs)
        else:
            x_list = self.decoderhead(backbone_outputs, gtmask)
        return x_list

"""
对于不满足该范式的模型可在backbone部分进行定义, 并在此处导入
"""

# model_config
def build_model(cfg):
    c = CDMaskFormer(cfg)
    return c
