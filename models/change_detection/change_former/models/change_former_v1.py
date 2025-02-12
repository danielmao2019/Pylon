from typing import Dict
import torch
from models.change_detection.change_former.modules.tenc import Tenc
from models.change_detection.change_former.modules.conv_projection_base import convprojection_base
from models.change_detection.change_former.modules.conv_layer import ConvLayer


class ChangeFormerV1(torch.nn.Module):

    def __init__(self, input_nc=3, output_nc=2):
        super(ChangeFormerV1, self).__init__()

        self.Tenc               = Tenc()
        self.convproj           = convprojection_base()
        self.change_probability = ConvLayer(8, output_nc, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        x1, x2 = inputs['img_1'], inputs['img_2']

        fx1 = self.Tenc(x1)
        fx2 = self.Tenc(x2)

        DI = []
        for i in range(0,4):
            DI.append(torch.abs(fx1[i] - fx2[i]))

        cp = self.convproj(DI)

        cp = self.change_probability(cp)

        return cp
