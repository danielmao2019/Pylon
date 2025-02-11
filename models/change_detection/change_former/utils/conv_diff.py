import torch


def conv_diff(in_channels, out_channels):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        torch.nn.ReLU()
    )
