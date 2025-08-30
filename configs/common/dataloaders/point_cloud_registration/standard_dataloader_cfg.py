import torch
from data.collators.base_collator import BaseCollator


dataloader_cfg = {
    'class': torch.utils.data.DataLoader,
    'args': {
        'batch_size': 1,
        'num_workers': 4,
        'collate_fn': {
            'class': BaseCollator,
            'args': {},
        },
    },
}
