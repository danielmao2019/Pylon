from typing import Dict
import torch
from models.multi_task_learning.backbones import UNetEncoder, UNetDecoder


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

        if arch == 'FC-Siam-conc':
            # For concatenation, adjust the input features
            upconv_in_features = [2 * self.encoder.out_features[-1]] + self.encoder.out_features[-2:-5:-1]
            dec_in_features = list(map(lambda x: 3*x, self.encoder.out_features[-2::-1]))
        else:
            upconv_in_features = self.encoder.out_features[-1:-5:-1]
            dec_in_features = list(map(lambda x: 2*x, self.encoder.out_features[-2::-1]))

        self.decoder = UNetDecoder(
            upconv_in_features=upconv_in_features,
            dec_in_features=dec_in_features,
            out_features=self.encoder.out_features[-2::-1],
            num_classes=num_classes,
        )

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

    def _forward_FC_Siam_conc(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        x1, x2 = inputs['img_1'], inputs['img_2']
        x1, x2 = self.encoder(x1), self.encoder(x2)
        conc = list(map(
            lambda x: torch.cat([x[0], x[1]], dim=1),
            list(zip(x1, x2)),
        ))
        x = self.decoder(conc)
        return x

    def _forward_FC_Siam_diff(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        x1, x2 = inputs['img_1'], inputs['img_2']
        x1, x2 = self.encoder(x1), self.encoder(x2)
        diff = list(map(lambda x: torch.abs(x[0]-x[1]), list(zip(x1, x2))))
        x = self.decoder(diff)
        return x
