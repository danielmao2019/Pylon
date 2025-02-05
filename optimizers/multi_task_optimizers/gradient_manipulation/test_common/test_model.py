from typing import Tuple, List, Dict
from functools import partial
import torch


def save_activations(
    activations: Dict[str, torch.Tensor],
    name: str,
    module: torch.nn.Module,
    inp: Tuple,
    out: torch.Tensor
) -> None:
    """PyTorch Forward hook to save outputs at each forward
    pass. Mutates specified dict objects with each fwd pass.
    """
    if isinstance(module, (torch.nn.Linear, torch.nn.ReLU)):
        assert type(out) == torch.Tensor
        activations[name] = out


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
        self.activations: Dict[str, torch.Tensor] = {}
        for name, module in self.named_modules():
            module.register_forward_hook(
                partial(save_activations, self.activations, name)
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_rep = self.backbone(x)
        outputs: Dict[str, torch.Tensor] = {
            name: self.heads[name](shared_rep)
            for name in self.heads
        }
        outputs['shared_rep'] = shared_rep
        return outputs
