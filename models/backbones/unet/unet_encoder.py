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
        assert isinstance(out_features, list)
        assert all(isinstance(x, int) for x in out_features)

        self.out_features = out_features

        # Create contracting path layers dynamically
        self.encoders = torch.nn.ModuleList([unet_block(
            in_channels=in_channels if i == 0 else out_features[i - 1],
            out_channels=out_features[i],
        ) for i in range(len(out_features))])
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        skip_connections: List[torch.Tensor] = []

        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x) if encoder != self.encoders[-1] else x

        return skip_connections
