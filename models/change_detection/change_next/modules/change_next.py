import torch.nn as nn
import numpy as np
import math
from timm.models.layers import to_2tuple, to_3tuple
import torch
import collections
import torch.nn.functional as F
from models.ChangeFormer import *
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mmcv.cnn.bricks import DropPath
from models.Decoder import ChangeNeXtDecoder
import torch.nn as nn
import torch
from mmcv.cnn import build_conv_layer, build_norm_layer, build_activation_layer
from mmcv.runner import BaseModule
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
# from mmseg.models.decode_heads.decode_head import BaseDecodeHead,BaseDecodeHeadNew
from mmseg.ops import resize
from torch import Tensor
import math
from mmseg.models.utils import SelfAttentionBlock
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
from models.ChangeFormer import EncoderTransformer_v3
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class StemConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StemConv, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels // 2,
                      kernel_size=(3, 3),
                      stride=(2, 2),
                      padding=(1, 1)),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2,
                      out_channels,
                      kernel_size=(3, 3),
                      stride=(2, 2),
                      padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class MSCAN(nn.Module):
    def __init__(self,
                 in_chans=3,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 depths=[3, 4, 6, 3],
                 num_stages=4):
        super(MSCAN, self).__init__()

        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
               ]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(3, embed_dims[0])
            else:
                patch_embed = OverlapPatchEmbed(
                    patch_size=7 if i == 0 else 3,
                    stride=4 if i == 0 else 2,
                    in_chans=in_chans if i == 0 else embed_dims[i - 1],
                    embed_dim=embed_dims[i])

            block = nn.ModuleList([
                Block(dim=embed_dims[i],
                      mlp_ratio=mlp_ratios[i],
                      drop=drop_rate,
                      drop_path=dpr[cur + j]) for j in range(depths[i])
            ])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def init_weights(self, pretrained=None):
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    nn.init.normal_(m,
                                    mean=0,
                                    std=math.sqrt(2.0 / fan_out),
                                    bias=0)
        elif isinstance(pretrained, str):
            self.load_parameters(torch.load(pretrained))

    def forward(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            # print(H.device)
            for blk in block:
                x = blk(x, H, W)
            # print(x.device)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            outs.append(x)

        return outs
