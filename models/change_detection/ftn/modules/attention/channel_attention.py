import torch


class ChannelAttention(torch.nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.conv = torch.nn.Conv2d(3 * in_planes, in_planes, 1, 1, 0)
        self.bn = torch.nn.BatchNorm2d(in_planes)
        self.SiLU = torch.nn.SiLU()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)

        self.fc = torch.nn.Sequential(torch.nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                torch.nn.SiLU(),
                                torch.nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = torch.nn.Sigmoid()
        self.bnnorm = torch.nn.BatchNorm2d(in_planes)

    def forward(self, x):
        x = self.conv(x)  # 2-1
        x = self.bn(x)
        x = self.SiLU(x)
        res = x
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        attn = self.sigmoid(out)
        out = x * attn + res
        return out


class ChannelAttention_1(torch.nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention_1, self).__init__()
        self.conv = torch.nn.Conv2d(2 * in_planes, in_planes, 3, 1, 1)
        self.bn = torch.nn.BatchNorm2d(in_planes)
        self.SiLU = torch.nn.SiLU()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)

        self.fc = torch.nn.Sequential(torch.nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                torch.nn.SiLU(),
                                torch.nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.SiLU(x)
        res = x
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        result = x * self.sigmoid(out) + res
        return result
