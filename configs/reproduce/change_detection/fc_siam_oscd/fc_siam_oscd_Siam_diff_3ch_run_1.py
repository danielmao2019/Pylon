# This file is automatically generated by `./configs/reproduce/change_detection/generators/gen_fc_siam_oscd.py`.
# Please do not attempt to modify manually.
import torch
import data
import models
import criteria
import metrics
import optimizers
import runners


collate_fn_cfg = {
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
        'class': data.dataloaders.BaseDataLoader,
        'args': {
            'batch_size': 32,
            'last_mode': "fill",
            'num_workers': 4,
            'collate_fn': collate_fn_cfg,
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
config['train_dataset']['args']['bands'] = "3ch"
config['val_dataset']['args']['bands'] = "3ch"

# model config
config['model']['args']['arch'] = "FC-Siam-diff"
config['model']['args']['in_channels'] = 3

# seeds
config['init_seed'] = 59337984
config['train_seeds'] = [85957019, 33691820, 77210660, 310303, 52142976, 4364870, 88746313, 62171931, 35970040, 24566182, 65272382, 95245600, 34047276, 91644478, 8122245, 75339024, 46371748, 87827131, 14533920, 67143322, 76991284, 6807328, 21920383, 74770539, 59627589, 17356859, 64508536, 91028848, 29430622, 19042892, 25640744, 9493730, 73827393, 81691335, 53947123, 66327846, 43624342, 69732362, 98737238, 47149448, 80577627, 53221563, 60291970, 56481248, 5838001, 27989423, 10165823, 65106112, 11824444, 34673865, 17045881, 37124095, 43908790, 3582825, 10566130, 26971114, 91180276, 57294187, 82014837, 57755655, 84659522, 34577708, 72458874, 98215353, 14828032, 48230156, 14927527, 57948767, 86204650, 23126182, 28085932, 37207075, 5079901, 47746339, 77321490, 56720044, 35364463, 55427015, 93405012, 88501752, 75698985, 90704676, 49372725, 91223675, 4690623, 36686154, 49348149, 82272642, 14747898, 39311729, 80963750, 83494643, 99173590, 52313332, 6150646, 19476737, 34880247, 82805078, 41548290, 73824315]

# work dir
config['work_dir'] = "./logs/reproduce/change_detection/fc_siam_oscd/fc_siam_oscd_Siam_diff_3ch_run_1"
