from typing import List, Dict, Union
import torch
from models.change_detection.change_former.modules.encoder_transformer_x2 import EncoderTransformer_x2
from models.change_detection.change_former.modules.decoder_transformer_x2 import DecoderTransformer_x2


class ChangeFormerV4(torch.nn.Module):

    def __init__(self, input_nc=3, output_nc=2):
        super(ChangeFormerV4, self).__init__()
        #Transformer Encoder
        self.embed_dims = [32, 64, 128, 320, 512]
        self.depths     = [3, 3, 4, 12, 3] #[3, 3, 6, 18, 3]
        self.embedding_dim = 256

        self.Tenc_x2 = EncoderTransformer_x2(img_size=256, patch_size=3, in_chans=input_nc, num_classes=output_nc, embed_dims=self.embed_dims,
                 num_heads=[2, 2, 4, 8, 16], mlp_ratios=[2, 2, 2, 2, 2], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=torch.nn.LayerNorm,
                 depths=self.depths, sr_ratios=[8, 4, 2, 1, 1])

        #Transformer Decoder
        self.TDec_x2 = DecoderTransformer_x2(input_transform='multiple_select', in_index=[0, 1, 2, 3, 4], align_corners=True,
                    in_channels = self.embed_dims, embedding_dim= 256, output_nc=output_nc,
                    feature_strides=[2, 4, 8, 16, 32])

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Union[torch.Tensor, List[torch.Tensor]]:
        x1, x2 = inputs['img_1'], inputs['img_2']

        [fx1, fx2] = [self.Tenc_x2(x1), self.Tenc_x2(x2)]

        cp = self.TDec_x2(fx1, fx2)

        if not self.training:
            cp = cp[-1]

        return cp
