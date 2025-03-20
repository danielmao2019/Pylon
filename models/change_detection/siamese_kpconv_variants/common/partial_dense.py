from typing import *
import torch

from models.change_detection.siamese_kpconv_variants.common.base_modules import BaseModule, MLP
from models.change_detection.siamese_kpconv_variants.common.interpolate import KNNInterpolate


class FPModule_PD(BaseModule):
    """Upsampling module from PointNet++
    Arguments:
        k [int] -- number of nearest neighboors used for the interpolation
        up_conv_nn [List[int]] -- list of feature sizes for the uplconv mlp
    """

    def __init__(self, up_k, up_conv_nn, *args, **kwargs):
        super(FPModule_PD, self).__init__()
        self.upsample_op = KNNInterpolate(up_k)
        bn_momentum = kwargs.get("bn_momentum", 0.1)
        self.nn = MLP(up_conv_nn, bn_momentum=bn_momentum, bias=False)

    def forward(self, data, data_skip):
        batch_out = data_skip.copy()
        x_skip = data_skip['x']
        precomputed = data.get('multiscale', None)

        has_innermost = len(data['x']) == data['batch'].max() + 1

        if precomputed and not has_innermost:
            if 'up_idx' not in data:
                batch_out['up_idx'] = 0
            else:
                batch_out['up_idx'] = data['up_idx']

            pre_data = precomputed[batch_out['up_idx']]
            batch_out['up_idx'] = batch_out['up_idx'] + 1
        else:
            pre_data = None

        if has_innermost:
            x = torch.gather(data['x'], 0, data_skip['batch'].unsqueeze(-1).repeat((1, data['x'].shape[-1])))
        else:
            x = self.upsample_op(data, data_skip, precomputed=pre_data)

        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)

        if hasattr(self, "nn"):
            batch_out['x'] = self.nn(x)
        else:
            batch_out['x'] = x
        return batch_out
