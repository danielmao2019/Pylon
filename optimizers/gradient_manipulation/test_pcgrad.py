from typing import Dict, List
import pytest
from .pcgrad import PCGradOptimizer
import torch
from utils.models import get_flattened_params


class Model(torch.nn.module):

    def __init__(self) -> None:
        super(Model, self).__init__()
        torch.manual_seed(0)
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(2, 2), torch.nn.Linear(2, 2),
        )
        self.heads = torch.nn.ModuleDict({
            'task1': torch.nn.Sequential(
                torch.nn.Linear(2, 2), torch.nn.Linear(2, 2),
            ),
            'task2': torch.nn.Sequential(
                torch.nn.Linear(2, 2), torch.nn.Linear(2, 2),
            ),
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared = self.backbone(x)
        result: Dict[str, torch.Tensor] = {
            name: self.heads[name](shared)
            for name in self.heads
        }
        return result
 

class Dataset(torch.utils.data.Dataset):

    def __init__(self) -> None:
        super(Dataset, self).__init__()

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> torch.Tensor:
        torch.manual_seed(idx)
        inputs = torch.randn(size=(2,))
        labels = {
            'task1': torch.randn(size=(2,)),
            'task2': torch.randn(size=(2,)),
        }
        dp = {
            'inputs': inputs,
            'labels': labels,
        }
        return dp


def test_pcgrad_optimizer():
    model = Model()
    optimizer = PCGradOptimizer(
        optimizer_config={
            'class': torch.optim.SGD,
            'args': {
                'params': model.parameters(),
                'lr': 1e-03,
            }
        }, wrt_rep=False, per_layer=False, 
    )
    dataset = Dataset()
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=2,
    )
    l1 = lambda x, y: ((x-y) ** 2).sum()
    criterion = lambda x, y: {
        name: l1(x[name], y[name])
        for name in ['task1', 'task2']
    }
    trajectory: List[torch.Tensor] = [get_flattened_params(model).detatch().clone()]
    for dp in dataloader:
        outputs = model(dp['inputs'])
        losses = criterion(outputs, dp['labels'])
        optimizer.zero_grad()
        optimizer.backward(losses=losses, shared_rep=None)
        optimizer.step()
        trajectory.append(get_flattened_params(model).detatch().clone())
    from .test_pcgrad_ground_truth import ground_truth
    assert len(trajectory) == len(ground_truth)
    assert all(
        torch.equal(trajectory[i], ground_truth[i])
        for i in range(len(trajectory))
    )
