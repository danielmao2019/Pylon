from functools import partial
import torch
from models.change_detection.change_former.modules.tenc import Tenc
from models.change_detection.change_former.modules.tdec_v2 import TDecV2


class ChangeFormerV3(torch.nn.Module):

    def __init__(self, input_nc=3, output_nc=2, decoder_softmax=False):
        super(ChangeFormerV3, self).__init__()
        #Transformer Encoder
        self.Tenc = Tenc(patch_size=16, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 4, 8],
                            mlp_ratios=[4, 4, 4, 4], qkv_bias=True, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                            depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)

        #Transformer Decoder
        self.TDec = TDecV2(input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=True,
                            in_channels = [64, 128, 320, 512], embedding_dim= 64, output_nc=output_nc,
                            decoder_softmax = decoder_softmax, feature_strides=[4, 8, 16, 32])

    def forward(self, x1, x2):

        fx1 = self.Tenc(x1)
        fx2 = self.Tenc(x2)

        cp = self.TDec(fx1, fx2)

        return cp
