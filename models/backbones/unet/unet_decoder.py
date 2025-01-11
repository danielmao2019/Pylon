from typing import List
import torch
from .utils import unet_block


class UNetDecoder(torch.nn.Module):

    def __init__(self, num_classes: int) -> None:
        super(UNetDecoder, self).__init__()

        # Expanding path layers
        self.upconv4 = self._upconv(512, 256)
        self.dec4 = unet_block(512, 256)
        self.upconv3 = self._upconv(256, 128)
        self.dec3 = unet_block(256, 128)
        self.upconv2 = self._upconv(128, 64)
        self.dec2 = unet_block(128, 64)
        self.final_conv = torch.nn.Conv2d(64, num_classes, kernel_size=1)

    def _upconv(self, in_channels, out_channels):
        return torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, skip_connections: List[torch.Tensor]) -> torch.Tensor:
        x7, x5, x3, x1 = skip_connections

        x = self.upconv4(x7)
        x = torch.cat([x, x5], dim=1)
        x = self.dec4(x)

        x = self.upconv3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)

        x = self.upconv2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec2(x)

        x = self.final_conv(x)
        return x
