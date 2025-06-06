import torch
import torchvision
import data
import data.collators.change_star_collator
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
                [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'lbl_1'), ('labels', 'lbl_2')]
            ),
            (
                {
                    'class': data.transforms.vision_2d.RandomRotation,
                    'args': {'choices': [0, 90, 180, 270]},
                },
                [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'lbl_1'), ('labels', 'lbl_2')]
            ),
            (
                {
                    'class': data.transforms.Randomize,
                    'args': {'transform': data.transforms.vision_2d.Flip(axis=-1), 'p': 0.5},
                },
                [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'lbl_1'), ('labels', 'lbl_2')]
            ),
            (
                {
                    'class': data.transforms.Randomize,
                    'args': {'transform': data.transforms.vision_2d.Flip(axis=-2), 'p': 0.5},
                },
                [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'lbl_1'), ('labels', 'lbl_2')]
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

collate_fn_cfg = {
    'class': data.collators.ChangeStarCollator,
    'args': {
        'method': "train",
    },
}

data_cfg = {
    'train_dataset': {
        'class': data.datasets.xView2Dataset,
        'args': {
            'data_root': "./data/datasets/soft_links/xView2",
            'split': "train",
            'transforms_cfg': transforms_cfg,
        },
    },
    'train_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 4,
            'num_workers': 4,
            'collate_fn': collate_fn_cfg,
        },
    },
    'criterion': {
        'class': criteria.vision_2d.change_detection.ChangeStarCriterion,
        'args': {},
    },
}
