import torch
import data
import criteria
from .transforms_cfg import transforms_cfg


collate_fn_config = {
    'class': data.collators.BaseCollator,
    'args': {
        'collators': {},
    },
}

source_dataset = data.datasets.Bi2SingleTemporal(
    source=data.datasets.SYSU_CD_Dataset(data_root="./data/datasets/soft_links/SYSU-CD", split="train"),
)

class_dist = torch.Tensor(data.datasets.SYSU_CD_Dataset.CLASS_DIST['train']).to(torch.float32)
num_classes = data.datasets.SYSU_CD_Dataset.NUM_CLASSES
class_weights = num_classes * (1/class_dist) / torch.sum(1/class_dist)

config = {
    'train_dataset': {
        'class': data.datasets.I3PEDataset,
        'args': {
            'source': source_dataset,
            'dataset_size': len(source_dataset),
            'exchange_ratio': 0.75,
            'transforms_cfg': transforms_cfg,
        },
    },
    'train_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 128,
            'num_workers': 8,
            'collate_fn': collate_fn_config,
        },
    },
    'criterion': {
        'class': criteria.vision_2d.SymmetricChangeDetectionCriterion,
        'args': {
            'class_weights': tuple(class_weights.tolist()),
        },
    },
}
