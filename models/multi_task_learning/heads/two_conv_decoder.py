import torch


class TwoConvDecoder(torch.nn.Module):

    def __init__(self, out_channels: int, in_channels: int = 64):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels,
            kernel_size=3, padding=1,
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=1, padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))
