from typing import List, Dict, Union
from functools import partial
import torch
from models.change_detection.change_former.modules.encoder_transformer_v3 import EncoderTransformer_v3
from models.change_detection.change_next.modules.decoder import ChangeNeXtDecoder


class ChangeNextV2(torch.nn.Module):
    def __init__(self, input_nc=3, output_nc=2, embed_dim=256, embed_dims=[64, 128, 320, 512], decoder_softmax=False,
                 mlp_ratios=[8, 8, 4, 4], drop_rate=0.1,
                 drop_path_rate=0.2, depths=[3, 3, 4, 3], num_stages=4,
                 ):
        super(ChangeNextV2, self).__init__()

        self.embedding_dim = embed_dim
        self.embed_dims = embed_dims
        self.drop_rate = drop_rate
        self.attn_drop = 0.1
        self.drop_path_rate = drop_path_rate
        self.depths = depths
        self.Tenc_x2 = EncoderTransformer_v3(img_size=256, patch_size=7, in_chans=input_nc, num_classes=output_nc,
                                             embed_dims=self.embed_dims,
                                             num_heads=[1, 2, 4, 8], mlp_ratios=mlp_ratios, qkv_bias=True,
                                             qk_scale=None, drop_rate=self.drop_rate,
                                             attn_drop_rate=self.attn_drop, drop_path_rate=self.drop_path_rate,
                                             norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                                             depths=self.depths, sr_ratios=[8, 4, 2, 1])

        self.Decode = ChangeNeXtDecoder(interpolate_mode='bilinear', num_heads=4, m=0.9, trans_with_mlp=True,
                                        trans_depth=1,
                                        att_type="XCA", in_channels=self.embed_dims, in_index=[0, 1, 2, 3],
                                        channels=embed_dim,
                                        dropout_ratio=0.1, num_classes=2, input_transform='multiple_select',
                                        align_corners=False, feature_strides=[2, 4, 8, 16],
                                        embedding_dim=self.embedding_dim, output_nc=output_nc,
                                        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Union[torch.Tensor, List[torch.Tensor]]:
        x1, x2 = inputs['img_1'], inputs['img_2']
        [fx1, fx2] = [self.Tenc_x2(x1), self.Tenc_x2(x2)]

        cp = self.Decode(fx1, fx2)

        if not self.training:
            cp = cp[-1]
        return cp
