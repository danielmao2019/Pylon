import torch


class CSA_CDGAN_Discriminator(torch.nn.Module):

    def __init__(self, isize, nc, nz, ndf, n_extra_layers=0):
        super(CSA_CDGAN_Discriminator, self).__init__()
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
        self.toplayer = torch.nn.Conv2d(ndf*4, nz, 3, 1, 1, bias=False)
        self.avgpool = torch.torch.nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.e1(x)
        x = self.e_extra_layers(x)
        x = self.e2(x)
        x = self.e3(x)
        x = self.toplayer(x)
        x = self.avgpool(x)
        x = x.squeeze(3).squeeze(2)
        assert x.ndim == 2
        return x
