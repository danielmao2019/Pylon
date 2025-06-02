import torch
from data.datasets.random_datasets import BaseRandomDataset
from data.transforms import Compose


torch.manual_seed(0)
gt = torch.rand(size=(10, 10), dtype=torch.float32)

def gt_func(x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    return gt @ x + noise * 0.001


dataset_cfg = {
    'class': BaseRandomDataset,
    'args': {
        'num_examples': 100,
        'initial_seed': 0,
        'gen_func_config': {
            'inputs': {
                'x': (
                    torch.rand,
                    {'size': (10,), 'dtype': torch.float32},
                ),
            },
            'labels': {
                'y': (
                    torch.randn,
                    {'size': (10,), 'dtype': torch.float32},
                ),
            },
        },
        'transforms_cfg': {
            'class': Compose,
            'args': {
                'transforms': [
                    {
                        'op': gt_func,
                        'input_names': [('inputs', 'x'), ('labels', 'y')],
                        'output_names': [('labels', 'y')],
                    }
                ],
            },
        },
    },
}
