import torch
from data.datasets.random_datasets import BaseRandomDataset
from data.transforms import Compose


# Global device setting
device = torch.device('cuda')

torch.manual_seed(0)
gt = torch.rand(size=(2, 2), dtype=torch.float32, device=device)


def gt_func(x: torch.Tensor, noise: torch.Tensor, seed=None) -> torch.Tensor:
    return gt @ x + noise * 0.001


dataset_cfg = {
    'class': BaseRandomDataset,
    'args': {
        'num_examples': 10,
        'base_seed': 0,
        'device': device,
        'use_cpu_cache': False,
        'use_disk_cache': False,
        'gen_func_config': {
            'inputs': {
                'x': (
                    torch.rand,
                    {'size': (2,), 'dtype': torch.float32},
                ),
            },
            'labels': {
                'y': (
                    torch.randn,
                    {'size': (2,), 'dtype': torch.float32},
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
