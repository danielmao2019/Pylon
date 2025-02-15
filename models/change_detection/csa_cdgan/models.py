from typing import Dict, Optional
import torch
from .generator import CSA_CDGAN_Generator
from .discriminator import CSA_CDGAN_Discriminator


class CSA_CDGAN(torch.nn.Module):

    def __init__(self, isize, nc, nz, ndf, n_extra_layers, num_classes: Optional[int] = 2) -> None:
        super(CSA_CDGAN, self).__init__()
        self.generator = CSA_CDGAN_Generator(isize, nc, nz, ndf, n_extra_layers, num_classes)
        self.discriminator = CSA_CDGAN_Discriminator(isize, nc, nz, ndf, n_extra_layers)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert not self.training
        return self.generator(inputs)
