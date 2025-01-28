from typing import Tuple, Dict, Union
import torch


class SNUNet(torch.nn.Module):

    def __init__(self, encoder: torch.nn.Module, change_decoder_cfg: dict, semantic_decoder_cfg: dict) -> None:
        super(SNUNet, self).__init__()

    # ====================================================================================================
    # initialization methods
    # ====================================================================================================

    @staticmethod
    def _build_change_decoder(
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        drop_rate: float,
        scale_factor: Union[float, Tuple[float, float]],
        num_convs: int,
    ) -> torch.nn.Module:
        in_layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, mid_channels, 3, 1, 1),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(mid_channels),
            DropConnect(drop_rate) if drop_rate > 0. else torch.nn.Identity()
        )
        mid_layers = [torch.nn.Sequential(
            torch.nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(mid_channels),
            DropConnect(drop_rate) if drop_rate > 0. else torch.nn.Identity()
        ) for _ in range(num_convs - 1)]
        out_layer = torch.nn.Sequential(
            torch.nn.Conv2d(mid_channels, out_channels, 3, 1, 1),
            torch.nn.UpsamplingBilinear2d(scale_factor=scale_factor),
        )
        return torch.nn.Sequential(in_layer, *mid_layers, out_layer)

    @staticmethod
    def _build_semantic_decoder(
        in_channels: int,
        out_channels: int,
        scale_factor: Union[float, Tuple[float, float]],
    ) -> torch.nn.Module:
        layers = [
            torch.nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            torch.nn.UpsamplingBilinear2d(scale_factor=scale_factor),
        ]
        return torch.nn.Sequential(*layers)

    # ====================================================================================================
    # forward methods
    # ====================================================================================================

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        feat_1 = self.encoder(inputs['img_1'])
        feat_2 = self.encoder(inputs['img_2'])
        if self.training:
            return self._forward_train(feat_1, feat_2)
        else:
            return self._forward_eval(feat_1, feat_2)

    def _forward_train(self, feat_1: torch.Tensor, feat_2: torch.Tensor) -> Dict[str, torch.Tensor]:
        change_12 = self.change_decoder(torch.cat([feat_1, feat_2], dim=1))
        change_21 = self.change_decoder(torch.cat([feat_2, feat_1], dim=1))
        semantic = self.semantic_decoder(feat_1)
        return {
            'change_12': change_12,
            'change_21': change_21,
            'semantic': semantic,
        }

    def _forward_eval(self, feat_1: torch.Tensor, feat_2: torch.Tensor) -> Dict[str, torch.Tensor]:
        change = self.change_decoder(torch.cat([feat_1, feat_2], dim=1))
        semantic_1 = self.semantic_decoder(feat_1)
        semantic_2 = self.semantic_decoder(feat_2)
        return {
            'change': change,
            'semantic_1': semantic_1,
            'semantic_2': semantic_2,
        }
