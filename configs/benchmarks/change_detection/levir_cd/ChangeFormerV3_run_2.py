# This file is automatically generated by `./configs/benchmarks/change_detection/gen_bi_temporal.py`.
# Please do not attempt to modify manually.
import torch
import optimizers


config = {
    'runner': None,
    'work_dir': None,
    'epochs': 100,
    # seeds
    'init_seed': None,
    'train_seeds': None,
    # dataset config
    'train_dataset': None,
    'train_dataloader': None,
    'criterion': None,
    'val_dataset': None,
    'val_dataloader': None,
    'test_dataset': None,
    'test_dataloader': None,
    'metric': None,
    # model config
    'model': None,
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

from runners import SupervisedSingleTaskTrainer
config['runner'] = SupervisedSingleTaskTrainer

# dataset config
from configs.common.datasets.change_detection.train.levir_cd import config as train_dataset_config
config.update(train_dataset_config)
from configs.common.datasets.change_detection.val.levir_cd import config as val_dataset_config
config.update(val_dataset_config)

# model config
import models
config['model'] = {'class': models.change_detection.ChangeFormerV3, 'args': {}}

from configs.common.datasets.change_detection.train._transforms_cfg import transforms_cfg
config['train_dataset']['args']['transforms_cfg'] = transforms_cfg(size=(256, 256))
config['val_dataset']['args']['transforms_cfg'] = transforms_cfg(size=(256, 256))

# criterion config
import criteria
config['criterion']['class'] = criteria.vision_2d.change_detection.ChangeFormerCriterion

# seeds
config['init_seed'] = 28896403
config['train_seeds'] = [81613822, 65533439, 17892967, 76120045, 84676015, 76215137, 43432514, 21776681, 35974420, 89325725, 17724019, 30262536, 56120685, 26428061, 76160310, 26342669, 49486639, 91040373, 95226301, 49914856, 30385925, 87478411, 64524560, 39023160, 91730606, 12104851, 96279430, 14754082, 89019192, 24575571, 80961308, 56646548, 72547875, 62194247, 84069638, 75772368, 8917510, 8131200, 84230472, 39071136, 40133123, 69951392, 16441706, 72944697, 97312856, 45010115, 70042381, 93771573, 30009779, 88260317, 33183654, 62259463, 93143524, 85399184, 67072476, 5990883, 7274266, 49420415, 29775954, 2639375, 47092333, 45064227, 8731119, 26132908, 83667469, 68435265, 85962279, 13787219, 76660613, 13804919, 94834228, 33093226, 48404259, 31717741, 19135151, 86326193, 61315409, 16238214, 14969274, 48738142, 48454350, 73230196, 89765677, 21735698, 93006474, 62380281, 9572639, 69087299, 57720133, 39417171, 9236298, 67994327, 44594009, 29987666, 59919008, 45994258, 60600148, 45125955, 52915304, 58978465]

# work dir
config['work_dir'] = "./logs/benchmarks/change_detection/levir_cd/ChangeFormerV3_run_2"
