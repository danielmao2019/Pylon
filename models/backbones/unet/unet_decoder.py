from typing import List, Optional
import torch
from .utils import unet_block


class UNetDecoder(torch.nn.Module):

    def __init__(
        self,
        in_features: Optional[int] = [1024, 512, 256, 128, 64],
        num_classes: int = None,
    ) -> None:
        super(UNetDecoder, self).__init__()
        assert type(in_features) == list
        assert all(type(x) == int for x in in_features)
        assert type(num_classes) == int, f"{type(num_classes)=}"
        # Expanding path layers
        self.upconv5 = self._upconv(in_features[0], in_features[1])
        self.dec5 = unet_block(in_features[0], in_features[1])
        self.upconv4 = self._upconv(in_features[1], in_features[2])
        self.dec4 = unet_block(in_features[1], in_features[2])
        self.upconv3 = self._upconv(in_features[2], in_features[3])
        self.dec3 = unet_block(in_features[2], in_features[3])
        self.upconv2 = self._upconv(in_features[3], in_features[4])
        self.dec2 = unet_block(in_features[3], in_features[4])
        self.final_conv = torch.nn.Conv2d(in_features[4], num_classes, kernel_size=1)

    def _upconv(self, in_channels, out_channels):
        return torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, skip_connections: List[torch.Tensor]) -> torch.Tensor:
        x9, x7, x5, x3, x1 = skip_connections

        x = self.upconv5(x9)
        x = torch.cat([x, x7], dim=1)
        x = self.dec5(x)

        x = self.upconv4(x)
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
