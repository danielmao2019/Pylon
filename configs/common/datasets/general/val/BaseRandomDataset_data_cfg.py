from data.datasets.random_datasets import BaseRandomDataset

data_cfg = {
    'val_dataset': {
        'class': BaseRandomDataset,
        'args': {
            'num_examples': 16,
            'initial_seed': 42,
            'device': 'cpu',
            'gen_func_config': {
                'inputs': {
                    'x': (
                        'torch.rand',
                        {'size': (2,), 'dtype': 'torch.float32'},
                    ),
                },
                'labels': {
                    'y': (
                        'torch.rand',
                        {'size': (1,), 'dtype': 'torch.float32'},
                    ),
                },
            },
            'transforms_cfg': None,
        }
    }
}