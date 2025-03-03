import torch
import optimizers
import runners


collate_fn_config = {
    'class': data.collators.BaseCollator,
    'args': {
        'collators': {
            'meta_info': {
                'date_1': list,
                'date_2': list,
            },
        },
    },
}

class_dist = torch.Tensor(data.datasets.OSCDDataset.CLASS_DIST['train']).to(torch.float32)
num_classes = data.datasets.OSCDDataset.NUM_CLASSES
class_weights = num_classes * (1/class_dist) / torch.sum(1/class_dist)

config = {
    'runner': runners.SupervisedSingleTaskTrainer,
    'work_dir': None,
    'epochs': 100,
    # seeds
    'init_seed': None,
    'train_seeds': None,
    # dataset config
    'train_dataset': {
        'class': data.datasets.OSCDDataset,
        'args': {
            'data_root': "./data/datasets/soft_links/OSCD",
            'split': "train",
            'transforms_cfg': {
                'class': data.transforms.Compose,
                'args': {
                    'transforms': [
                        (
                            data.transforms.crop.RandomCrop(size=(96, 96), interpolation=None),
                            [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'change_map')],
                        ),
                        (
                            data.transforms.RandomRotation(choices=[0, 90, 180, 270]),
                            [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'change_map')],
                        ),
                        (
                            data.transforms.Randomize(transform=data.transforms.Flip(axis=-1), p=0.5),
                            [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'change_map')],
                        ),
                        (
                            data.transforms.Randomize(transform=data.transforms.Flip(axis=-2), p=0.5),
                            [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'change_map')],
                        ),
                    ],
                },
            },
            'bands': None,
        },
    },
    'train_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 32,
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
    'val_dataset': {
        'class': data.datasets.OSCDDataset,
        'args': {
            'data_root': "./data/datasets/soft_links/OSCD",
            'split': "test",
            'transforms_cfg': {
                'class': data.transforms.Compose,
                'args': {
                    'transforms': [
                        (
                            data.transforms.crop.RandomCrop(size=(96, 96), interpolation=None),
                            [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'change_map')],
                        ),
                    ],
                },
            },
            'bands': None,
        },
    },
    'val_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 1,
            'num_workers': 4,
            'collate_fn': collate_fn_cfg,
        },
    },
    'test_dataset': None,
    'test_dataloader': None,
    'metric': {
        'class': metrics.vision_2d.SemanticSegmentationMetric,
        'args': {
            'num_classes': 2,
        },
    },
    # model config
    'model': {
        'class': models.change_detection.FullyConvolutionalSiameseNetwork,
        'args': {
            'arch': None,
            'in_channels': None,
            'num_classes': 2,
        },
    },
    # optimizer config
    'optimizer': {
        'class': optimizers.SingleTaskOptimizer,
        'args': {
            'optimizer_config': {
                'class': torch.optim.SGD,
                'args': {
                    'lr': 1.0e-03,
                    'momentum': 0.9,
                    'weight_decay': 1.0e-04,
                },
            },
        },
    },
    # scheduler config
    'scheduler': {
        'class': torch.optim.lr_scheduler.PolynomialLR,
        'args': {
            'optimizer': None,
            'total_iters': None,
            'power': 0.9,
        },
    },
}
