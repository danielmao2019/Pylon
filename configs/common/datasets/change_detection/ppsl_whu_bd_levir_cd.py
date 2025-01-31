import criteria.vision_2d.ppsl_criterion
import torch
import data
import criteria
import metrics


train_transforms_cfg = {
    'class': data.transforms.Compose,
    'args': {
        'transforms': [
            (
                data.transforms.resize.ResizeMaps(size=(256, 256), antialias=True),
                [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'change_map'), ('labels', 'semantic_map')]
            ),
        ],
    },
}

eval_transforms_cfg = {
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

source_dataset = data.datasets.WHU_BD_Dataset(data_root="./data/datasets/soft_links/WHU-BD", split="train")

config = {
    'train_dataset': {
        'class': data.datasets.PPSLDataset,
        'args': {
            'source': source_dataset,
            'dataset_size': len(source_dataset),
            'transforms_cfg': train_transforms_cfg,
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
    'val_dataset': {
        'class': data.datasets.LevirCdDataset,
        'args': {
            'data_root': "./data/datasets/soft_links/LEVIR_CD",
            'split': "val",
            'transforms_cfg': eval_transforms_cfg,
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
        'class': data.datasets.LevirCdDataset,
        'args': {
            'data_root': "./data/datasets/soft_links/LEVIR_CD",
            'split': "test",
            'transforms_cfg': eval_transforms_cfg,
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
        'class': criteria.vision_2d.PPSLCriterion,
        'args': {},
    },
    'metric': {
        'class': metrics.vision_2d.SemanticSegmentationMetric,
        'args': {
            'num_classes': 2,
        },
    },
}
