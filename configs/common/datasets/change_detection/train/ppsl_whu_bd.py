import torch
import torchvision
import data
import criteria


transforms_cfg = {
    'class': data.transforms.Compose,
    'args': {
        'transforms': [
            (
                {
                    'class': data.transforms.vision_2d.RandomCrop,
                    'args': {'size': (224, 224)},
                },
                [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'change_map'), ('labels', 'semantic_map')]
            ),
            (
                {
                    'class': data.transforms.vision_2d.RandomRotation,
                    'args': {'choices': [0, 90, 180, 270]},
                },
                [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'change_map'), ('labels', 'semantic_map')]
            ),
            (
                {
                    'class': data.transforms.Randomize,
                    'args': {
                        'transform': data.transforms.vision_2d.Flip(axis=-1),
                        'p': 0.5,
                    },
                },
                [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'change_map'), ('labels', 'semantic_map')]
            ),
            (
                {
                    'class': data.transforms.Randomize,
                    'args': {
                        'transform': data.transforms.vision_2d.Flip(axis=-2),
                        'p': 0.5,
                    },
                },
                [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'change_map'), ('labels', 'semantic_map')]
            ),
            (
                {
                    'class': data.transforms.Randomize,
                    'args': {
                        'transform': torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                        'p': 0.5,
                    },
                },
                ('inputs', 'img_1'),
            ),
            (
                {
                    'class': data.transforms.Randomize,
                    'args': {
                        'transform': torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                        'p': 0.5,
                    },
                },
                ('inputs', 'img_2'),
            ),
        ],
    },
}

data_cfg = {
    'train_dataset': {
        'class': data.datasets.PPSLDataset,
        'args': {
            'source': {
                'class': data.datasets.WHU_BD_Dataset,
                'args': {
                    'data_root': "./data/datasets/soft_links/WHU-BD",
                    'split': "train",
                },
            },
            'dataset_size': data.datasets.WHU_BD_Dataset.DATASET_SIZE['train'],
            'transforms_cfg': transforms_cfg,
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
        'class': criteria.vision_2d.change_detection.PPSLCriterion,
        'args': {},
    },
}
