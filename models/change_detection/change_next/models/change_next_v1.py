from typing import Dict
import torch
from models.change_detection.change_next.modules.change_next import MSCAN
from models.change_detection.change_former.modules.decoder_transformer_v3 import DecoderTransformer_v3


class ChangeNextV1(torch.nn.Module):
    def __init__(self, input_nc=3, output_nc=2, embed_dim=256, embed_dims=[64, 128, 320, 512],
                 mlp_ratios=[8, 8, 4, 4], drop_rate=0.1,
                 drop_path_rate=0.2, depths=[3, 3, 12, 3], num_stages=4
                 ):
        super(ChangeNextV1, self).__init__()

        self.embedding_dim = embed_dim
        self.embed_dims = embed_dims

        self.Tenc_x2 = MSCAN(in_chans=input_nc,
                             embed_dims=embed_dims,
                             mlp_ratios=mlp_ratios,
                             drop_rate=drop_rate,
                             drop_path_rate=drop_path_rate,
                             depths=depths,
                             num_stages=num_stages)
        self.Decode = DecoderTransformer_v3(
            input_transform='multiple_select', in_index=[0, 1, 2, 3],
            align_corners=False,
            in_channels=self.embed_dims, embedding_dim=self.embedding_dim,
            output_nc=output_nc,
            feature_strides=[2, 4, 8, 16],
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        x1, x2 = inputs['img_1'], inputs['img_2']
        [fx1, fx2] = [self.Tenc_x2(x1), self.Tenc_x2(x2)]

        cp = self.Decode(fx1, fx2)

        return cp
