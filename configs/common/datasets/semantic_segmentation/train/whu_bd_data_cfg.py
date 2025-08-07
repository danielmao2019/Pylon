import torch
from data.datasets import WHU_BD_Dataset
from criteria.vision_2d import SemanticSegmentationCriterion


data_cfg = {
    'train_dataset': {
        'class': WHU_BD_Dataset,
        'args': {
            'data_root': './data/datasets/soft_links/WHU-BD',
            'split': 'train',
        },
    },
    'train_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 64,
            'num_workers': 4,
        },
    },
    'criterion': {
        'class': SemanticSegmentationCriterion,
        'args': {},
    },
}
