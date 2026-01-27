from typing import List

import torch

from models.multi_task_learning.backbones.unet.utils import unet_block


class UNetDecoder(torch.nn.Module):

    def __init__(
        self,
        upconv_in_features: List[int],
        dec_in_features: List[int],
        out_features: List[int],
        num_classes: int,
    ) -> None:
        # Input validations
        assert isinstance(
            upconv_in_features, list
        ), "upconv_in_features must be a list."
        assert all(
            isinstance(value, int) for value in upconv_in_features
        ), "upconv_in_features must contain only int values."
        assert isinstance(dec_in_features, list), "dec_in_features must be a list."
        assert all(
            isinstance(value, int) for value in dec_in_features
        ), "dec_in_features must contain only int values."
        assert isinstance(out_features, list), "out_features must be a list."
        assert all(
            isinstance(value, int) for value in out_features
        ), "out_features must contain only int values."
        assert len(out_features) > 0, "out_features must contain at least one value."
        assert (
            len(upconv_in_features) == len(dec_in_features) == len(out_features)
        ), "upconv_in_features, dec_in_features, and out_features must have matching lengths."
        assert isinstance(num_classes, int), "num_classes must be an int."

        super(UNetDecoder, self).__init__()

        self.upconvs = torch.nn.ModuleList(
            [
                self._upconv(up_in, out)
                for up_in, out in zip(upconv_in_features, out_features, strict=True)
            ]
        )
        self.decoders = torch.nn.ModuleList(
            [
                unet_block(dec_in, out)
                for dec_in, out in zip(dec_in_features, out_features, strict=True)
            ]
        )
        self.final_conv = torch.nn.Conv2d(
            in_channels=out_features[-1], out_channels=num_classes, kernel_size=1
        )

    def _upconv(self, in_channels: int, out_channels: int) -> torch.nn.ConvTranspose2d:
        return torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
        )

    def forward(self, skip_connections: List[torch.Tensor]) -> torch.Tensor:
        # Input validations
        assert isinstance(skip_connections, list), "skip_connections must be a list."
        assert (
            len(skip_connections) == len(self.upconvs) + 1
        ), "skip_connections length must be one more than the number of upconvs."
        assert all(
            isinstance(item, torch.Tensor) for item in skip_connections
        ), "skip_connections must contain only torch.Tensor values."

        skip_connections = skip_connections[::-1]
        x = skip_connections[0]
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            x = torch.cat([x, skip_connections[i + 1]], dim=1)
            x = self.decoders[i](x)
        x = self.final_conv(x)
        return x
