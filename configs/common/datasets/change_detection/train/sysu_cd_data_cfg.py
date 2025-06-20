import torch
import data
import criteria
from configs.common.datasets.change_detection.train._transforms_cfg import transforms_cfg


collate_fn_config = {
    'class': data.collators.BaseCollator,
    'args': {
        'collators': {},
    },
}

class_dist = torch.Tensor(data.datasets.SYSU_CD_Dataset.CLASS_DIST['train']).to(torch.float32)
num_classes = data.datasets.SYSU_CD_Dataset.NUM_CLASSES
class_weights = num_classes * (1/class_dist) / torch.sum(1/class_dist)

data_cfg = {
    'train_dataset': {
        'class': data.datasets.SYSU_CD_Dataset,
        'args': {
            'data_root': "./data/datasets/soft_links/SYSU-CD",
            'split': "train",
            'transforms_cfg': transforms_cfg(size=(224, 224)),
        },
    },
    'train_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 4,
            'num_workers': 4,
            'collate_fn': collate_fn_config,
        },
    },
    'criterion': {
        'class': criteria.vision_2d.SemanticSegmentationCriterion,
        'args': {
            'class_weights': tuple(class_weights.tolist()),
        },
    },
}
