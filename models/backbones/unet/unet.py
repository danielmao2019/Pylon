import torch
from .unet_encoder import UNetEncoder
from .unet_decoder import UNetDecoder


class UNet(torch.nn.Module):

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super(UNet, self).__init__()
        self.encoder = UNetEncoder(in_channels=in_channels)
        self.decoder = UNetDecoder(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
