from typing import Dict
import torch
from models.change_detection.cdx_former.modules.sea_former import SeaFormer_L
from models.change_detection.cdx_former.modules.cdx_lstm import CDXLSTM


class CDXFormer(torch.nn.Module):

    def __init__(self) -> None:
        self.encoder = SeaFormer_L(pretrained=True)
        self.decoder = CDXLSTM([64, 128, 192, 256])

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.decoder(self.encoder(inputs['img_1'], inputs['img_2']))
