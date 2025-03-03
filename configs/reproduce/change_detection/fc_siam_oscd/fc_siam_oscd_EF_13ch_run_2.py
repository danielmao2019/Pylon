# This file is automatically generated by `./configs/reproduce/change_detection/generators/gen_fc_siam_oscd.py`.
# Please do not attempt to modify manually.
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

# dataset config
config['train_dataset']['args']['bands'] = 13ch
config['val_dataset']['args']['bands'] = 13ch

# model config
config['model']['args']['arch'] = "FC-EF"
config['model']['args']['in_channels'] = 26

# seeds
config['init_seed'] = 81565104
config['train_seeds'] = [27762215, 57945845, 64083695, 67828048, 91413715, 53381123, 41780800, 71051537, 46258869, 12660385, 11917690, 28194322, 58524994, 74652921, 97389511, 79859017, 77340092, 56144712, 97826565, 8109674, 43169794, 1583931, 14627731, 53809643, 95423711, 4458785, 3454073, 45272773, 69197155, 23816465, 10618412, 38792207, 77858580, 52838255, 56353741, 66567119, 77858225, 2902129, 642771, 53386270, 69903742, 51402692, 51603249, 63825042, 63188361, 57272108, 26516117, 84338144, 7331095, 48914447, 58806922, 76528962, 62456997, 89507576, 68929330, 45623711, 1087565, 46737600, 30306152, 14690958, 90725522, 56975094, 81083331, 79503596, 680708, 44056087, 65557691, 94457845, 37848019, 87968937, 97604957, 23939931, 82914112, 82099993, 17779339, 47007786, 31776497, 71655726, 97055774, 64686502, 50865812, 25447385, 33231069, 98554343, 17483362, 65361950, 16801986, 16676536, 93819958, 98104136, 65117942, 74009119, 98079322, 96675071, 8623646, 14095528, 126911, 75404197, 62668996, 56360900]

# work dir
config['work_dir'] = "./logs/reproduce/change_detection/fc_siam_oscd/fc_siam_oscd_EF_13ch_run_2"
