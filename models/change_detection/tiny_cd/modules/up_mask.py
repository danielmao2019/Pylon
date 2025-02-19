from typing import Optional
import torch


class UpMask(torch.nn.Module):

    def __init__(
        self,
        scale_factor: float,
        nin: int,
        nout: int,
    ):
        super(UpMask, self).__init__()
        self._upsample = torch.nn.Upsample(
            scale_factor=scale_factor, mode="bilinear", align_corners=True
        )
        self._convolution = torch.nn.Sequential(
            torch.nn.Conv2d(nin, nin, 3, 1, groups=nin, padding=1),
            torch.nn.PReLU(),
            torch.nn.InstanceNorm2d(nin),
            torch.nn.Conv2d(nin, nout, kernel_size=1, stride=1),
            torch.nn.PReLU(),
            torch.nn.InstanceNorm2d(nout),
        )

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self._upsample(x)
        if y is not None:
            x = x * y
        return self._convolution(x)
