import torch
from models.change_detection.csa_cdgan.modules import constants as ct
from models.change_detection.csa_cdgan.modules import attention as at


class CSA_CDGAN_Generator(torch.nn.Module):

    def __init__(self, isize, nc, nz, ndf, n_extra_layers=0):
        super(CSA_CDGAN_Generator, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        self.e1 = torch.nn.Sequential(
            torch.nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ndf),
            torch.nn.ReLU(True),
            )
        self.e_extra_layers = torch.nn.Sequential()
        for t in range(n_extra_layers):
            self.e_extra_layers.add_module('extra-layers-{0}-{1}-conv'.format(t, ndf),
                            torch.nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False))
            self.e_extra_layers.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, ndf),
                            torch.nn.BatchNorm2d(ndf))
            self.e_extra_layers.add_module('extra-layers-{0}-{1}-relu'.format(t, ndf),
                            torch.nn.LeakyReLU(0.2, inplace=True))
        self.e2 = torch.nn.Sequential(
            torch.nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ndf*2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            )
        self.e3 = torch.nn.Sequential(
            torch.nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ndf*4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            )
        self.e4 = torch.nn.Sequential(
            torch.nn.Conv2d(ndf*4, nz, 3, 1, 1, bias=False),
            )
        self.d4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(nz, ndf*4, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(ndf*4),
            torch.nn.ReLU(True),
            )
        self.d3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(ndf*4*2, ndf*2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ndf*2),
            torch.nn.ReLU(True),
            )
        self.d2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(ndf*4, ndf, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ndf),
            torch.nn.ReLU(True),
            )
        self.d_extra_layers = torch.nn.Sequential()
        for t in range(n_extra_layers):
            self.d_extra_layers.add_module('extra-layers-{0}-{1}-conv'.format(ndf*2, ndf),
                            torch.nn.Conv2d(ndf*2, ndf, 3, 1, 1, bias=False))
            self.d_extra_layers.add_module('extra-layers-{0}-{1}-batchnorm'.format(ndf, ndf),
                            torch.nn.BatchNorm2d(ndf))
            self.d_extra_layers.add_module('extra-layers-{0}-{1}-relu'.format(ndf, ndf),
                            torch.nn.ReLU(inplace=True))
        self.d1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(ndf*2, ct.GT_C, 4, 2, 1, bias=False),
            torch.nn.Sigmoid(),
            )
        #attention module

        self.at1 = at.csa_layer(1)
        self.at2 = at.csa_layer(1)
        self.at3 = at.csa_layer(1)
        self.at4 = at.csa_layer(1)
        
    def forward(self,x):
        
        e1 = self.e1(x)
        e_el = self.e_extra_layers(e1)
        e2 = self.e2(e_el)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        d4 = self.d4(e4)
        
        d4 = self.at4(d4)
        
        c34 = torch.cat((e3,d4),1)
        d3 = self.d3(c34)
        d3 = self.at3(d3)
        
        c23 = torch.cat((e2,d3),1)
        d2 = self.d2(c23)
        d2 = self.at2(d2)
        
        cel2 = torch.cat((e_el,d2),1)
        d_el = self.d_extra_layers(cel2)
        e_el = self.at1(d_el)
        
        c11 = torch.cat((e1,d_el),1)
        d1 = self.d1(c11)
        
        return d1
