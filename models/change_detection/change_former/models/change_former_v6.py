from typing import List, Dict, Union
from functools import partial
import torch
from models.change_detection.change_former.modules.encoder_transformer_v3 import EncoderTransformer_v3
from models.change_detection.change_former.modules.decoder_transformer_v3 import DecoderTransformer_v3


class ChangeFormerV6(torch.nn.Module):

    def __init__(self, input_nc=3, output_nc=2, embed_dim=256):
        super(ChangeFormerV6, self).__init__()
        #Transformer Encoder
        self.embed_dims = [64, 128, 320, 512]
        self.depths     = [3, 3, 4, 3] #[3, 3, 6, 18, 3]
        self.embedding_dim = embed_dim
        self.drop_rate = 0.1
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1

        self.Tenc_x2 = EncoderTransformer_v3(img_size=256, patch_size = 7, in_chans=input_nc, num_classes=output_nc, embed_dims=self.embed_dims,
                 num_heads = [1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=self.drop_rate,
                 attn_drop_rate = self.attn_drop, drop_path_rate=self.drop_path_rate, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                 depths=self.depths, sr_ratios=[8, 4, 2, 1])

        #Transformer Decoder
        self.TDec_x2 = DecoderTransformer_v3(input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=False,
                    in_channels = self.embed_dims, embedding_dim= self.embedding_dim, output_nc=output_nc,
                    feature_strides=[2, 4, 8, 16])

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Union[torch.Tensor, List[torch.Tensor]]:
        x1, x2 = inputs['img_1'], inputs['img_2']

        [fx1, fx2] = [self.Tenc_x2(x1), self.Tenc_x2(x2)]

        cp = self.TDec_x2(fx1, fx2)
        assert type(cp) == list, f"{type(cp)=}"

        if not self.training:
            cp = cp[-1]

        return cp
