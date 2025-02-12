import torch
import data
import criteria
from ._transforms_cfg import transforms_cfg


collate_fn_config = {
    'class': data.collators.BaseCollator,
    'args': {
        'collators': {
            'meta_info': {
                'image_resolution': torch.Tensor,
            },
        },
    },
}

class_dist = torch.Tensor(data.datasets.LevirCdDataset.CLASS_DIST['train']).to(torch.float32)
num_classes = data.datasets.LevirCdDataset.NUM_CLASSES
class_weights = num_classes * (1/class_dist) / torch.sum(1/class_dist)

config = {
    'train_dataset': {
        'class': data.datasets.LevirCdDataset,
        'args': {
            'data_root': "./data/datasets/soft_links/LEVIR-CD",
            'split': "train",
            'transforms_cfg': transforms_cfg((224, 224)),
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
