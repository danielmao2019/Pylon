from typing import Dict, List
import pytest
from .pcgrad import PCGradOptimizer
import torch
from utils.models import get_flattened_params


class Model(torch.nn.Module):

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
        shared_rep = self.backbone(x)
        outputs: Dict[str, torch.Tensor] = {
            name: self.heads[name](shared_rep)
            for name in self.heads
        }
        outputs['shared_rep'] = shared_rep
        return outputs


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
    dataset = Dataset()
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=2,
    )
    l1 = lambda x, y: ((x-y) ** 2).sum()
    criterion = lambda x, y: {
        name: l1(x[name], y[name])
        for name in ['task1', 'task2']
    }
    dp = next(iter(dataloader))
    example_outputs = model(dp['inputs'])
    example_losses = criterion(example_outputs, dp['labels'])
    optimizer = PCGradOptimizer(
        optimizer_config={
            'class': torch.optim.SGD,
            'args': {
                'params': model.parameters(),
                'lr': 1e-03,
            }
        }, losses=example_losses, shared_rep=example_outputs['shared_rep'],
        wrt_rep=False, per_layer=False, 
    )
    trajectory: List[torch.Tensor] = [get_flattened_params(model).detach().clone()]
    # for dp in dataloader:
    #     outputs = model(dp['inputs'])
    #     losses = criterion(outputs, dp['labels'])
    #     optimizer.zero_grad()
    #     optimizer.backward(losses=losses, shared_rep=None)
    #     optimizer.step()
    #     trajectory.append(get_flattened_params(model).detach().clone())
    from .test_pcgrad_ground_truth import ground_truth
    assert len(trajectory) == len(ground_truth)
    assert all(
        torch.all(torch.isclose(trajectory[i], ground_truth[i], rtol=0, atol=1e-04))
        for i in range(len(trajectory))
    )
