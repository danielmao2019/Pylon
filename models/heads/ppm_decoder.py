from typing import Tuple, Optional
import torch


class PyramidPoolingModule(torch.nn.Module):

    DIM: int = 512

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        pool_scales: Optional[Tuple[int]] = (1, 2, 3, 6),
    ) -> None:
        super(PyramidPoolingModule, self).__init__()
        # init ppm
        self.ppm = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(scale),
            torch.nn.Conv2d(
                in_channels=in_channels, out_channels=self.DIM, kernel_size=1, bias=False,
            ),
            torch.nn.BatchNorm2d(self.DIM),
            torch.nn.ReLU(inplace=True),
        ) for scale in pool_scales])
        # init conv_last
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels+len(pool_scales)*self.DIM, out_channels=self.DIM,
                kernel_size=3, padding=1, bias=False,
            ),
            torch.nn.BatchNorm2d(self.DIM),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(self.DIM, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # apply all pooling modules
        x = torch.cat([x] + [
            torch.nn.functional.upsample(pool(x), (x.shape[2], x.shape[3]), mode="bilinear")
            for pool in self.ppm
        ], dim=1)
        # apply conv
        x = self.conv(x)
        return x
