import torch
from models.change_detection.ftn.modules.conv import CrossConv


class ppattention_wan(torch.nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.conv = CrossConv(2 * in_planes, in_planes)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
        # b,h,w,c   ---  n*c   #n,,c,h,w   ---- n*c
        self.fc = torch.nn.Sequential(torch.nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                torch.nn.SiLU(),
                                torch.nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.fc2 = torch.nn.Conv2d(1, 1, 1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()
        self.bnnorm = torch.nn.BatchNorm2d(in_planes)

    def forward(self, x):
        x = self.conv(x)  # 2-->1,CROSS CONV
        res = x
        avg_out = self.fc(self.avg_pool(x))
        # max_out = self.fc(self.max_pool(x))
        # out = avg_out + max_out
        attn = self.sigmoid(avg_out)
        x_channel_summation = torch.sum(x, dim=1, keepdim=True)
        attn_channel_summation = self.sigmoid(self.fc2(x_channel_summation))
        result = x * attn + res + attn_channel_summation * res
        return result


class DFE(torch.nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.fc = torch.nn.Sequential(torch.nn.Conv2d(in_planes, in_planes // 2, 1, bias=False),
                                torch.nn.BatchNorm2d(in_planes // 2),
                                torch.nn.SiLU())

    def forward(self, x):
        result = self.fc(x)
        return result
