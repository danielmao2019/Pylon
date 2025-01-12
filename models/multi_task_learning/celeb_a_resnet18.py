from typing import Dict, Optional
import torch
import models
from data.datasets import CelebADataset


class CelebA_ResNet18(torch.nn.Module):

    def __init__(self, return_shared_rep: Optional[bool] = True) -> None:
        super(CelebA_ResNet18, self).__init__()
        self.return_shared_rep = return_shared_rep
        # initialize backbone
        self.backbone = models.backbones.resnet18(weights='DEFAULT')
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        # initialize decoders
        self.decoders = torch.nn.ModuleDict({
            task: torch.nn.Linear(in_features=512, out_features=2)
            for task in CelebADataset.LABEL_NAMES[1:]
        })

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = inputs['image']
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        shared_rep = x
        outputs: Dict[str, torch.Tensor] = {
            task: self.decoders[task](shared_rep)
            for task in self.decoders
        }
        if self.return_shared_rep:
            outputs['shared_rep'] = shared_rep
        return outputs
