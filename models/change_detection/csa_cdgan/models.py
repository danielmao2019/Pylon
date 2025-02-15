from typing import Dict
import torch
import utils


class CSA_CDGAN(torch.nn.Module):

    def __init__(self, generator_cfg: dict, discriminator_cfg: dict) -> None:
        super(CSA_CDGAN, self).__init__()
        self.generator = utils.builders.build_from_config(generator_cfg)
        self.discriminator = utils.builders.build_from_config(discriminator_cfg)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert not self.training
        return self.generator(inputs)
