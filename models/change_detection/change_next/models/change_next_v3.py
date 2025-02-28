from typing import Dict
import torch
from models.change_detection.change_next.modules.change_next import MSCAN
from models.change_detection.change_next.modules.decoder import ChangeNeXtDecoder


class ChangeNextV3(torch.nn.Module):
    def __init__(self, input_nc=3, output_nc=2, embed_dim=256, embed_dims=[64, 128, 320, 512],
                 mlp_ratios=[8, 8, 4, 4], drop_rate=0.,
                 drop_path_rate=0.1, depths=[3, 3, 12, 3], num_stages=4
                 ):
        super(ChangeNextV3, self).__init__()

        self.embedding_dim = embed_dim
        self.embed_dims = embed_dims

        self.Tenc_x2 = MSCAN(in_chans=input_nc,
                             embed_dims=embed_dims,
                             mlp_ratios=mlp_ratios,
                             drop_rate=drop_rate,
                             drop_path_rate=drop_path_rate,
                             depths=depths,
                             num_stages=num_stages)
        self.Decode = ChangeNeXtDecoder(interpolate_mode='bilinear', num_heads=4, m=0.9, trans_with_mlp=True,
                                        trans_depth=64,
                                        att_type="XCA", in_channels=self.embed_dims, in_index=[0, 1, 2, 3],
                                        channels=embed_dim,
                                        dropout_ratio=0.1, num_classes=2, input_transform='multiple_select',
                                        align_corners=False, feature_strides=[2, 4, 8, 16],
                                        embedding_dim=self.embedding_dim, output_nc=output_nc,
                                        decoder_softmax=False)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        x1, x2 = inputs['img_1'], inputs['img_2']
        [fx1, fx2] = [self.Tenc_x2(x1), self.Tenc_x2(x2)]

        cp = self.Decode(fx1, fx2)

        return cp
