from typing import Dict
import torch
from utils.builders.builder import build_from_config


class Changer(torch.nn.Module):
    __doc__ = r"""
    Download model checkpoint:
    ```bash
    cd models/change_detection/changer
    mkdir -p checkpoints
    wget https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth
    wget https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b1_20220624-02e5a6a1.pth
    """

    def __init__(self, encoder_cfg, decoder_cfg) -> None:
        super(Changer, self).__init__()
        self.encoder = build_from_config(encoder_cfg)
        self.decoder = build_from_config(decoder_cfg)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        feats = self.encoder(inputs['img_1'], inputs['img_2'])
        assert isinstance(feats, list), f"{type(feats)=}"
        assert len(feats) == 4
        assert all(isinstance(x, torch.Tensor) for x in feats)
        out = self.decoder(feats)
        assert isinstance(out, torch.Tensor)
        return out
