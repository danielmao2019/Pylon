from typing import Dict, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import build_backbone
from .decoder import build_decoder
from .utils import CBAM, DS_layer


class DSAMNet(nn.Module):
    def __init__(
        self,
        n_class: int = 2,
        ratio: int = 8,
        kernel: int = 7,
        backbone: str = 'resnet18',
        output_stride: int = 16,
        f_c: int = 64,
        freeze_bn: bool = False,
        in_c: int = 3,
    ) -> None:
        super(DSAMNet, self).__init__()
        BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm, in_c)
        self.decoder = build_decoder(f_c, BatchNorm)

        self.cbam0 = CBAM(f_c, ratio, kernel)
        self.cbam1 = CBAM(f_c, ratio, kernel)

        self.ds_lyr2 = DS_layer(64, 32, 2, 1, n_class)
        self.ds_lyr3 = DS_layer(128, 32, 4, 3, n_class)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]:
        input1, input2 = inputs['image1'], inputs['image2']
        x_1, f2_1, f3_1, f4_1 = self.backbone(input1)
        x_2, f2_2, f3_2, f4_2 = self.backbone(input2)

        x1 = self.decoder(x_1, f2_1, f3_1, f4_1)
        x2 = self.decoder(x_2, f2_2, f3_2, f4_2)

        x1 = self.cbam0(x1). transpose(1,3) 
        x2 = self.cbam1(x2). transpose(1,3)  # channel = 64

        dist = F.pairwise_distance(x1, x2, keepdim=True).transpose(1, 3)  # channel = 1
        dist = F.interpolate(dist, size=input1.shape[2:], mode='bilinear', align_corners=True)

        ds2 = self.ds_lyr2(torch.abs(f2_1 - f2_2))
        ds3 = self.ds_lyr3(torch.abs(f3_1 - f3_2))

        if self.training:
            return dist.squeeze(1), ds2, ds3
        else:
            return torch.cat([2 - dist, dist], dim=1)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
