import torch
import torchvision
import data
import criteria
import metrics


transforms_config = {
    'class': data.transforms.Compose,
    'args': {
        'transforms': [
            (
                torchvision.transforms.Resize(size=(256, 256), antialias=True),
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
    'val_dataset': {
        'class': data.datasets.SYSU_CD_Dataset,
        'args': {
            'data_root': "./data/datasets/soft_links/SYSU-CD",
            'split': "val",
            'transforms_cfg': transforms_config,
        },
    },
    'val_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 1,
            'num_workers': 4,
            'collate_fn': collate_fn_config,
        },
    },
    'test_dataset': {
        'class': data.datasets.SYSU_CD_Dataset,
        'args': {
            'data_root': "./data/datasets/soft_links/SYSU-CD",
            'split': "test",
            'transforms_cfg': transforms_config,
        },
    },
    'test_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 1,
            'num_workers': 4,
            'collate_fn': collate_fn_config,
        },
    },
    'criterion': {
        'class': criteria.vision_2d.SemanticSegmentationCriterion,
        'args': {
            'class_weights': tuple((
                data.datasets.SYSU_CD_Dataset.NUM_CLASSES*(1/torch.tensor(data.datasets.SYSU_CD_Dataset.CLASS_DIST['train']))/torch.sum(1/torch.tensor(data.datasets.SYSU_CD_Dataset.CLASS_DIST['train']))
            ).tolist()),
        },
    },
    'metric': {
        'class': metrics.vision_2d.SemanticSegmentationMetric,
        'args': {
            'num_classes': 2,
        },
    },
}
