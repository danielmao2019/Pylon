import torch
from models.change_detection.ftn.utils import autopad


class Conv(torch.nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = torch.nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = torch.nn.BatchNorm2d(c2)
        self.act = torch.nn.SiLU() if act is True else (act if isinstance(act, torch.nn.Module) else torch.nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class CrossConv(torch.nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_)  # , (1, k), (1, s))
        self.cv2 = Conv(c_, c2)  # , (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
