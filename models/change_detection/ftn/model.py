import torch
from models.change_detection.ftn.modules.swin.encoder import encoder1
from models.change_detection.ftn.modules.swin.swin_trans_decoder import SwinTransDecoder


class FTN(torch.nn.Module):
    def __init__(self):
        super(FTN, self).__init__()

        self.encoder1 = encoder1()
        self.decoder = SwinTransDecoder()

    def forward(self, img1, img2):
        x, x_downsample1, x_downsample2 = self.encoder1(img1, img2)
        x_p, x_2, x_3, x_4 = self.decoder(x, x_downsample1, x_downsample2)

        return x_p, x_2, x_3, x_4
