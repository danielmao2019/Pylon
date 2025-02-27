from typing import Dict
import torch
from utils.builders.builder import build_from_config


class Changer(torch.nn.Module):

    def __init__(self, encoder_cfg, decoder_cfg) -> None:
        self.encoder = build_from_config(encoder_cfg)
        self.decoder = build_from_config(decoder_cfg)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        feats = self.encoder(inputs['img_1'], inputs['img_2'])
        assert isinstance(feats, list)
        assert len(feats) == 4
        assert all(isinstance(x, torch.Tensor) for x in feats)
        out = self.decoder(feats)
        assert isinstance(out, torch.Tensor)
        return out
