import torch
import data
import criteria


transforms_config = {
    'class': data.transforms.Compose,
    'args': {
        'transforms': [
            (
                data.transforms.resize.ResizeMaps(size=(256, 256), antialias=True),
                [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'change_map')]
            ),
        ],
    },
}

collate_fn_config = {
    'class': data.collators.BaseCollator,
    'args': {
        'collators': {},
    },
}

class_dist = torch.Tensor(data.datasets.SYSU_CD_Dataset.CLASS_DIST['train'], dtype=torch.float32)
num_classes = data.datasets.SYSU_CD_Dataset.NUM_CLASSES
class_weights = num_classes * (1/class_dist) / torch.sum(1/class_dist)

config = {
    'train_dataset': {
        'class': data.datasets.SYSU_CD_Dataset,
        'args': {
            'data_root': "./data/datasets/soft_links/SYSU-CD",
            'split': "train",
            'transforms_cfg': transforms_config,
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
