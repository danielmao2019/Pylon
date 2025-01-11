from typing import Dict
import torch
from models.backbones import UNetEncoder, UNetDecoder


class FullyConvolutionalSiameseNetwork(torch.nn.Module):
    __doc__ = r"""
    References:
        * https://github.com/rcdaudt/fully_convolutional_change_detection
        * https://github.com/kyoukuntaro/FCSN_for_ChangeDetection_IGARSS2018

    Used in:

    """

    def __init__(self, arch: str, in_channels: int, num_classes: int) -> None:
        super(FullyConvolutionalSiameseNetwork, self).__init__()
        assert arch in ['FC-EF', 'FC-Siam-conc', 'FC-Siam-diff']
        self.arch = arch
        self.encoder = UNetEncoder(in_channels=in_channels)
        self.decoder = UNetDecoder(num_classes=num_classes)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert type(inputs) == dict and set(inputs.keys()) == set(['img_1', 'img_2'])
        if self.arch == 'FC-EF':
            return self._forward_FC_EF(inputs)
        if self.arch == 'FC-Siam-conc':
            return self._forward_FC_Siam_conc(inputs)
        if self.arch == 'FC-Siam-diff':
            return self._forward_FC_Siam_diff(inputs)

    def _forward_FC_EF(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat([inputs['img_1'], inputs['img_2']], dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x
