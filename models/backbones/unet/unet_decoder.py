from typing import List, Optional
import torch
from .utils import unet_block


class UNetDecoder(torch.nn.Module):

    def __init__(
        self,
        upconv_in_features: Optional[List[int]],
        dec_in_features: Optional[List[int]],
        out_features: Optional[List[int]],
        num_classes: int = None,
    ) -> None:
        super(UNetDecoder, self).__init__()
        assert len(upconv_in_features) == len(dec_in_features) == len(out_features)
        assert type(num_classes) == int, f"{type(num_classes)=}"

        self.upconvs = torch.nn.ModuleList(
            [self._upconv(up_in, out) for up_in, out in zip(upconv_in_features, out_features)]
        )
        self.decoders = torch.nn.ModuleList(
            [unet_block(dec_in, out) for dec_in, out in zip(dec_in_features, out_features)]
        )
        self.final_conv = torch.nn.Conv2d(out_features[-1], num_classes, kernel_size=1)

    def _upconv(self, in_channels, out_channels):
        return torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, skip_connections):
        x = skip_connections[0]
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            x = torch.cat([x, skip_connections[i + 1]], dim=1)
            x = self.decoders[i](x)
        x = self.final_conv(x)
        return x
