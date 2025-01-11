from typing import List, Optional
import torch
from .utils import unet_block


class UNetEncoder(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_features: Optional[List[int]] = [64, 128, 256, 512, 1024],
    ) -> None:
        super(UNetEncoder, self).__init__()
        assert type(in_channels) == int, f"{type(in_channels)=}"
        assert type(out_features) == list
        assert all(type(x) == int for x in out_features)
        self.out_features = out_features
        # Contracting path layers
        self.enc1 = unet_block(in_channels, out_features[0])
        self.enc2 = unet_block(out_features[0], out_features[1])
        self.enc3 = unet_block(out_features[1], out_features[2])
        self.enc4 = unet_block(out_features[2], out_features[3])
        self.enc5 = unet_block(out_features[3], out_features[4])
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
        x8 = self.pool(x7)

        x9 = self.enc5(x8)
        skip_connections.append(x9)

        return skip_connections
