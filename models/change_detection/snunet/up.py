import torch

class up(torch.nn.Module):
    __doc__ = r"""
    References:
        * https://github.com/likyoo/Siam-NestedUNet/blob/master/models/Models.py
    """
    
    def __init__(self, in_ch, bilinear=False):
        super(up, self).__init__()

        if bilinear:
            self.up = torch.nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
        else:
            self.up = torch.nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x):

        x = self.up(x)
        return x
