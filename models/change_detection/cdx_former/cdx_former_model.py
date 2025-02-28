from typing import Dict
import torch
from models.change_detection.cdx_former.modules.sea_former import SeaFormer_L
from models.change_detection.cdx_former.modules.cdx_lstm import CDXLSTM


class CDXFormer(torch.nn.Module):
    __doc__ = r"""
    Backbone pretrained weights download: https://drive.google.com/drive/folders/1BrZU0339JAFpKsQf4kdS0EpeeFgrBvBJ?usp=drive_link
    """

    def __init__(self) -> None:
        super(CDXFormer, self).__init__()
        self.encoder = SeaFormer_L(pretrained=True)
        self.decoder = CDXLSTM([64, 128, 192, 256])

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.decoder([
            self.encoder(inputs['img_1']),
            self.encoder(inputs['img_2']),
        ])
