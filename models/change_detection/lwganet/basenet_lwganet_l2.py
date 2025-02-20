import torch

from .backbone import LWGANet_L2_1242_e96_k11_RELU
from .neighbor_feature_aggregation import NeighborFeatureAggregation
from .temporal_fusion import TemporalFusionModule
from .decoder import Decoder

class BaseNet_LWGANet_L2(torch.nn.Module):
    def __init__(self, pretrained=True, preptrained_path=None):
        super().__init__()
        self.backbone = LWGANet_L2_1242_e96_k11_RELU(pretrained=pretrained, pretrained_path=preptrained_path)
        channles = [96, 96, 192, 384, 768]
        self.en_d = 32
        self.mid_d = self.en_d * 2
        self.swa = NeighborFeatureAggregation(channles, self.mid_d)
        self.tfm = TemporalFusionModule(self.mid_d, self.en_d * 2)
        self.decoder = Decoder(self.en_d * 2)

    def forward(self, inputs) -> torch.Tensor:
        # forward backbone resnet
        x1_2, x1_3, x1_4, x1_5 = self.backbone(inputs['img_1'])
        x2_2, x2_3, x2_4, x2_5 = self.backbone(inputs['img_2'])
        
        #assert 0, f'{x1_2.shape}'
        # aggregation
        x1_2, x1_3, x1_4, x1_5 = self.swa(x1_2, x1_3, x1_4, x1_5)
        x2_2, x2_3, x2_4, x2_5 = self.swa(x2_2, x2_3, x2_4, x2_5)
        # temporal fusion
        c2, c3, c4, c5 = self.tfm(x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5)
        # fpn
        p2, p3, p4, p5, mask_p2, mask_p3, mask_p4, mask_p5 = self.decoder(c2, c3, c4, c5)

        # change map
        mask_p2 = torch.nn.functional.interpolate(mask_p2, scale_factor=(4, 4), mode='bilinear')
        mask_p2 = torch.sigmoid(mask_p2)
        mask_p3 = torch.nn.functional.interpolate(mask_p3, scale_factor=(8, 8), mode='bilinear')
        mask_p3 = torch.sigmoid(mask_p3)
        mask_p4 = torch.nn.functional.interpolate(mask_p4, scale_factor=(16, 16), mode='bilinear')
        mask_p4 = torch.sigmoid(mask_p4)
        mask_p5 = torch.nn.functional.interpolate(mask_p5, scale_factor=(32, 32), mode='bilinear')
        mask_p5 = torch.sigmoid(mask_p5)

        return mask_p2, mask_p3, mask_p4, mask_p5