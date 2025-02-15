from typing import List, Dict
import torch
from models.change_detection.ftn.modules.swin.encoder import encoder1
from models.change_detection.ftn.modules.swin.swin_trans_decoder import SwinTransDecoder


class FTN(torch.nn.Module):
    def __init__(self):
        super(FTN, self).__init__()

        self.encoder1 = encoder1()
        self.decoder = SwinTransDecoder()

    def forward(self, inputs: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        img1, img2 = inputs['img_1'], inputs['img_2']
        x, x_downsample1, x_downsample2 = self.encoder1(img1, img2)
        out = self.decoder(x, x_downsample1, x_downsample2)  # out = [x_p, x_2, x_3, x_4]
        assert isinstance(out, tuple)
        assert len(out) == 4
        assert all(isinstance(x, torch.Tensor) for x in out)
        if not self.training:
            out = out[0]
        return out
