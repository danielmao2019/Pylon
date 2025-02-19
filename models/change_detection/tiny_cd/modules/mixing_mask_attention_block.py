from typing import List
import torch
from models.change_detection.tiny_cd.modules.mixing_block import MixingBlock
from models.change_detection.tiny_cd.modules.pixelwise_linear import PixelwiseLinear


class MixingMaskAttentionBlock(torch.nn.Module):
    """use the grouped convolution to make a sort of attention"""

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        fin: List[int],
        fout: List[int],
        generate_masked: bool = False,
    ):
        super(MixingMaskAttentionBlock, self).__init__()
        self._mixing = MixingBlock(ch_in, ch_out)
        self._linear = PixelwiseLinear(fin, fout)
        self._final_normalization = torch.nn.InstanceNorm2d(ch_out) if generate_masked else None
        self._mixing_out = MixingBlock(ch_in, ch_out) if generate_masked else None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z_mix = self._mixing(x, y)
        z = self._linear(z_mix)
        z_mix_out = 0 if self._mixing_out is None else self._mixing_out(x, y)

        return (
            z
            if self._final_normalization is None
            else self._final_normalization(z_mix_out * z)
        )
