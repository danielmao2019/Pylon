import torch
from models.change_detection.change_former.modules.tenc import Tenc
from models.change_detection.change_former.modules.tdec import TDec


class ChangeFormerV2(torch.nn.Module):

    def __init__(self, input_nc=3, output_nc=2, decoder_softmax=False):
        super(ChangeFormerV2, self).__init__()
        #Transformer Encoder
        self.Tenc   = Tenc()
        
        #Transformer Decoder
        self.TDec   = TDec( input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=True, 
                            in_channels = [64, 128, 320, 512], embedding_dim= 32, output_nc=output_nc, 
                            decoder_softmax = decoder_softmax, feature_strides=[4, 8, 16, 32])
        #Final activation
        self.decoder_softmax = decoder_softmax
        self.output_activation = torch.nn.Softmax(dim=1)

    def forward(self, x1, x2):

        fx1 = self.Tenc(x1)
        fx2 = self.Tenc(x2)

        DI = []
        for i in range(0,4):
            DI.append(torch.abs(fx1[i] - fx2[i]))

        cp = self.TDec(DI)

        if self.decoder_softmax:
            cp = self.output_activation(cp)

        return cp
