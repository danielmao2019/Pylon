import torch
import data
import criteria
from configs.common.datasets.change_detection.train._transforms_cfg import transforms_cfg


class_dist = torch.Tensor(data.datasets.SYSU_CD_Dataset.CLASS_DIST['train']).to(torch.float32)
num_classes = data.datasets.SYSU_CD_Dataset.NUM_CLASSES
class_weights = num_classes * (1/class_dist) / torch.sum(1/class_dist)

data_cfg = {
    'train_dataset': {
        'class': data.datasets.I3PEDataset,
        'args': {
            'source': {
                'class': data.datasets.Bi2SingleTemporal,
                'args': {
                    'data_root': "./data/datasets/soft_links/SYSU-CD",
                    'split': "train",
                }
            },
            'dataset_size': data.datasets.Bi2SingleTemporal.DATASET_SIZE['train'],
            'exchange_ratio': 0.75,
            'transforms_cfg': transforms_cfg(size=(224, 224)),
        },
    },
    'train_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 128,
            'num_workers': 8,
            'collate_fn': {
                'class': data.collators.BaseCollator,
                'args': {
                    'collators': {},
                },
            },
        },
    },
    'criterion': {
        'class': criteria.vision_2d.change_detection.SymmetricChangeDetectionCriterion,
        'args': {
            'class_weights': tuple(class_weights.tolist()),
        },
    },
}
