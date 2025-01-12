from typing import Dict, Optional
import torch
import data
import models


class MultiMNIST_LeNet5(torch.nn.Module):

    def __init__(self, return_shared_rep: Optional[bool] = True) -> None:
        super(MultiMNIST_LeNet5, self).__init__()
        self.return_shared_rep = return_shared_rep
        # initialize backbone
        self.backbone = models.backbones.LeNet5()
        # initialize decoders
        self.decoders = torch.nn.ModuleDict({
            task: torch.nn.Sequential(
                torch.nn.Linear(in_features=400, out_features=120),
                torch.nn.Sigmoid(),
                torch.nn.Linear(in_features=120, out_features=84),
                torch.nn.Sigmoid(),
                torch.nn.Linear(in_features=84, out_features=data.datasets.MultiMNISTDataset.NUM_CLASSES),
            ) for task in data.datasets.MultiMNISTDataset.LABEL_NAMES
        })

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = inputs['image']
        x = self.backbone(x)
        x = torch.flatten(x, start_dim=1)
        shared_rep = x
        outputs: Dict[str, torch.Tensor] = {
            task: self.decoders[task](shared_rep)
            for task in self.decoders
        }
        if self.return_shared_rep:
            outputs['shared_rep'] = shared_rep
        return outputs
