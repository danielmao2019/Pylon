from typing import List, Dict
import torch
import torch.nn as nn
from mmseg.models.utils import nlc_to_nchw
from mmseg.models.backbones import MixVisionTransformer
from models.change_detection.changer.modules.interaction_layer import TwoIdentity
from utils.builders.builder import build_from_config


class IA_MixVisionTransformer(MixVisionTransformer):
    def __init__(self, 
                 interaction_cfg=(None, None, None, None), 
                 **kwargs):
        super().__init__(**kwargs)
        assert self.num_stages == len(interaction_cfg), \
            'The length of the `interaction_cfg` should be same as the `num_stages`.'
        # cross-correlation
        self.ccs = []
        for ia_cfg in interaction_cfg:
            if ia_cfg is None:
                ia_cfg = {'class': TwoIdentity, 'args': {}}
            self.ccs.append(build_from_config(ia_cfg))
        self.ccs = nn.ModuleList(self.ccs)
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        x1, x2 = inputs['img_1'], inputs['img_2']
        outs = []
        for i, layer in enumerate(self.layers):
            x1, hw_shape = layer[0](x1)
            x2, hw_shape = layer[0](x2)
            for block in layer[1]:
                x1 = block(x1, hw_shape)
                x2 = block(x2, hw_shape)
            x1 = layer[2](x1)
            x2 = layer[2](x2)

            x1 = nlc_to_nchw(x1, hw_shape)
            x2 = nlc_to_nchw(x2, hw_shape)

            x1, x2 = self.ccs[i](x1, x2)
            if i in self.out_indices:
                outs.append(torch.cat([x1, x2], dim=1))
        return outs
