import torch
from functools import partial
from models.change_detection.change_former.modules.encoder_transformer import EncoderTransformer


class Tenc(EncoderTransformer):

    def __init__(self, **kwargs):
        super(Tenc, self).__init__(
            patch_size=16, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
