from typing import List
import torch
from .utils import unet_block


class UNetEncoder(torch.nn.Module):

    def __init__(self, in_channels: int) -> None:
        super(UNetEncoder, self).__init__()

        # Contracting path layers
        self.enc1 = unet_block(in_channels, 64)
        self.enc2 = unet_block(64, 128)
        self.enc3 = unet_block(128, 256)
        self.enc4 = unet_block(256, 512)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        skip_connections: List[torch.Tensor] = []

        x1 = self.enc1(x)
        skip_connections.append(x1)
        x2 = self.pool(x1)

        x3 = self.enc2(x2)
        skip_connections.append(x3)
        x4 = self.pool(x3)

        x5 = self.enc3(x4)
        skip_connections.append(x5)
        x6 = self.pool(x5)

        x7 = self.enc4(x6)
        skip_connections.append(x7)

        return skip_connections
