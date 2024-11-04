from typing import Dict
import torch


class GradientManipulationTestModel(torch.nn.Module):

    def __init__(self) -> None:
        super(GradientManipulationTestModel, self).__init__()
        torch.manual_seed(0)
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(2, 2), torch.nn.ReLU(),
            torch.nn.Linear(2, 2), torch.nn.ReLU(),
        )
        self.heads = torch.nn.ModuleDict({
            'task1': torch.nn.Sequential(
                torch.nn.Linear(2, 2), torch.nn.ReLU(),
                torch.nn.Linear(2, 2), torch.nn.ReLU(),
            ),
            'task2': torch.nn.Sequential(
                torch.nn.Linear(2, 2), torch.nn.ReLU(),
                torch.nn.Linear(2, 2), torch.nn.ReLU(),
            ),
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_rep = self.backbone(x)
        outputs: Dict[str, torch.Tensor] = {
            name: self.heads[name](shared_rep)
            for name in self.heads
        }
        outputs['shared_rep'] = shared_rep
        return outputs
