import torch


def conv2d_bn(in_channels, out_channels):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
        torch.nn.PReLU(),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.Dropout(p=0.6),
    )
