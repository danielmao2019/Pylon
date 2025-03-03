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
                            data.transforms.crop.RandomCrop(size=size, resize=resize, interpolation=None),
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
                        (
                            data.transforms.Randomize(torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5), p=0.5),
                            ('inputs', 'img_1'),
                        ),
                        (
                            data.transforms.Randomize(torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5), p=0.5),
                            ('inputs', 'img_2'),
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
    'val_dataset': {
        'class': data.datasets.OSCDDataset,
        'args': {
            'data_root': "./data/datasets/soft_links/OSCD",
            'split': "test",
            'transforms_cfg': transforms_cfg(size=(224, 224)),
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
    'metric': {
        'class': metrics.vision_2d.SemanticSegmentationMetric,
        'args': {
            'num_classes': 2,
        },
    },
    'test_dataset': None,
    'test_dataloader': None,
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
config['train_dataset']['args']['bands'] = 3ch
config['val_dataset']['args']['bands'] = 3ch

# model config
config['model']['args']['arch'] = "FC-Siam-conc"
config['model']['args']['in_channels'] = 3

# seeds
config['init_seed'] = 68379617
config['train_seeds'] = [81110544, 28038892, 81156962, 6133255, 66272721, 8804360, 86608210, 49036237, 9115756, 35942208, 91254002, 88207418, 71247484, 64310341, 75711600, 51236736, 82143823, 84196808, 71453892, 89443542, 53097406, 67206181, 27167470, 79452491, 39686748, 29890755, 96852260, 51332702, 64495776, 93810093, 52716731, 3427463, 56967929, 41526124, 95574743, 35554648, 3924839, 65056723, 47374719, 78941438, 46239617, 74334577, 10371589, 79236973, 31797105, 72810181, 54172295, 39525880, 9268880, 38473005, 53799485, 91109470, 93740588, 81923922, 22451743, 16467096, 77605640, 94960654, 6182028, 62195868, 79728317, 3618248, 863335, 53936582, 70259447, 84380741, 70394729, 77269035, 62976827, 58346318, 1890839, 906601, 5567066, 23974990, 42329922, 62829819, 84721825, 93638090, 90473107, 96470471, 62170406, 52494325, 12869406, 60859089, 44836485, 11839908, 42859491, 28833018, 67731632, 81117227, 61838637, 77886617, 40237248, 74979860, 36451592, 50846594, 86515655, 57095135, 26669130, 44732875]

# work dir
config['work_dir'] = "./logs/reproduce/change_detection/fc_siam_oscd/fc_siam_oscd_Siam_conc_3ch_run_0"
