from typing import Dict
import pytest
from .mtl_optimizer import MTLOptimizer
import torch
from utils.logging import Logger


LABEL_NAMES = ['a', 'b', 'c']


class MTLOptimizerTestModel(torch.nn.Module):

    def __init__(self) -> None:
        super(MTLOptimizerTestModel, self).__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.ReLU(),
            torch.nn.Linear(2, 2),
            torch.nn.ReLU(),
        )
        self.heads = torch.nn.ModuleDict({
            name: torch.nn.Linear(2, 2)
            for name in LABEL_NAMES
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_rep = self.backbone(x)
        outputs: Dict[str, torch.Tensor] = {}
        for name in self.heads:
            outputs[name] = self.heads[name](shared_rep)
        outputs['shared_rep'] = shared_rep
        return outputs


@pytest.mark.parametrize("wrt_rep", [
    (True),
    (False),
])
def test_mtl_optimizer(wrt_rep: bool) -> None:
    # input checks
    assert type(wrt_rep) == bool, f"{type(wrt_rep)=}"
    # initialization
    model = MTLOptimizerTestModel().cuda()
    core_optimizer_config = {
        'class': torch.optim.SGD,
        'args': {
            'params': model.parameters(),
            'lr': 1e-03,
        },
    }
    inputs = torch.randn(size=(1, 2), dtype=torch.float32, device='cuda')
    labels = {name: torch.randn(size=(1, 2), dtype=torch.float32, device='cuda') for name in LABEL_NAMES}
    criterion = torch.nn.L1Loss()
    outputs = model(inputs)
    losses = {name: criterion(outputs[name], labels[name]) for name in LABEL_NAMES}
    shared_rep = outputs['shared_rep']
    logger = Logger(filepath=None)
    optimizer = MTLOptimizer(
        optimizer_config=core_optimizer_config, losses=losses, shared_rep=shared_rep, logger=logger,
    )
    # ====================================================================================================
    # _init_shared_params_mask_
    # ====================================================================================================
    assert torch.equal(
        optimizer.shared_params_mask,
        torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.bool, device='cuda'),
    ), f"{optimizer.shared_params_mask=}"
    # ====================================================================================================
    # _get_shared_params_
    # ====================================================================================================
    assert torch.equal(
        torch.cat([p.flatten() for p in optimizer._get_shared_params_()], dim=0),
        torch.cat([
            list(model.backbone.children())[0].weight.flatten(),
            list(model.backbone.children())[0].bias.flatten(),
            list(model.backbone.children())[2].weight.flatten(),
            list(model.backbone.children())[2].bias.flatten(),
        ], dim=0)
    )
    # ====================================================================================================
    # _init_shared_params_shapes_
    # ====================================================================================================
    assert torch.equal(
        torch.cat([torch.tensor(s, dtype=torch.int64) for s in optimizer.shared_params_shapes]),
        torch.tensor([2, 2, 2, 2, 2, 2], dtype=torch.int64),
    )
    # ====================================================================================================
    # _get_grads_all_tasks_
    # ====================================================================================================
    produced = optimizer._get_grads_all_tasks_(losses=losses, shared_rep=shared_rep, wrt_rep=wrt_rep)
    expected = {
        name: (
            torch.abs(produced[name] - expected[name]) * 
            
        for name in LABEL_NAMES
    }
    assert produced.keys() == expected.keys(), f"{produced.keys()=}, {expected.keys()=}"
    assert all(torch.equal(produced[name], expected[name]) for name in LABEL_NAMES)
