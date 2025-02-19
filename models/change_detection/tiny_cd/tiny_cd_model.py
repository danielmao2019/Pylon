from typing import List, Dict
import torch
from models.change_detection.tiny_cd.modules.mixing_block import MixingBlock
from models.change_detection.tiny_cd.modules.pixelwise_linear import PixelwiseLinear
from models.change_detection.tiny_cd.modules.mixing_mask_attention_block import MixingMaskAttentionBlock
from models.change_detection.tiny_cd.modules.up_mask import UpMask
from models.change_detection.tiny_cd.utils import get_backbone


class TinyCDModel(torch.nn.Module):

    def __init__(
        self,
        bkbn_name="efficientnet_b4",
        pretrained=True,
        output_layer_bkbn="3",
        freeze_backbone=False,
    ):
        super(TinyCDModel, self).__init__()

        # Load the pretrained backbone according to parameters:
        self._backbone = get_backbone(
            bkbn_name, pretrained, output_layer_bkbn, freeze_backbone
        )

        # Initialize mixing blocks:
        self._first_mix = MixingMaskAttentionBlock(6, 3, [3, 10, 5], [10, 5, 1])
        self._mixing_mask = torch.nn.ModuleList(
            [
                MixingMaskAttentionBlock(48, 24, [24, 12, 6], [12, 6, 1]),
                MixingMaskAttentionBlock(64, 32, [32, 16, 8], [16, 8, 1]),
                MixingBlock(112, 56),
            ]
        )

        # Initialize Upsampling blocks:
        self._up = torch.nn.ModuleList(
            [
                UpMask(2, 56, 64),
                UpMask(2, 64, 64),
                UpMask(2, 64, 32),
            ]
        )

        # Final classification layer:
        self._classify = PixelwiseLinear([32, 16, 8], [16, 8, 1], torch.nn.Sigmoid())

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        ref, test = inputs['img_1'], inputs['img_2']
        features = self._encode(ref, test)
        latents = self._decode(features)
        return self._classify(latents)

    def _encode(self, ref, test) -> List[torch.Tensor]:
        features = [self._first_mix(ref, test)]
        for num, layer in enumerate(self._backbone):
            ref, test = layer(ref), layer(test)
            if num != 0:
                features.append(self._mixing_mask[num - 1](ref, test))
        return features

    def _decode(self, features) -> torch.Tensor:
        upping = features[-1]
        for i, j in enumerate(range(-2, -5, -1)):
            upping = self._up[i](upping, features[j])
        return upping
