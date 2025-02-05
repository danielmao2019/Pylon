import torch
from .test_model import GradientManipulationTestModel
from .test_dataset import GradientManipulationTestDataset


def initialize(optimizer_class, **kwargs):
    model = GradientManipulationTestModel()
    dataset = GradientManipulationTestDataset()
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=2)
    l1 = lambda x, y: ((x-y) ** 2).sum()
    criterion = lambda x, y: {
        name: l1(x[name], y[name])
        for name in ['task1', 'task2']
    }
    dp = next(iter(dataloader))
    example_outputs = model(dp['inputs'])
    example_losses = criterion(example_outputs, dp['labels'])
    optimizer = optimizer_class(
        optimizer_config={
            'class': torch.optim.SGD,
            'args': {
                'params': model.parameters(),
                'lr': 1e-03,
            }
        },
        losses=example_losses,
        shared_rep=example_outputs['shared_rep'],
        **kwargs,
    )
    return {
        'model': model,
        'dataloader': dataloader,
        'criterion': criterion,
        'optimizer': optimizer,
    }
