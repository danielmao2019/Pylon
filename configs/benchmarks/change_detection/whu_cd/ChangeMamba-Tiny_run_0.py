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
from configs.common.datasets.change_detection.train.whu_cd import config as train_dataset_config
config.update(train_dataset_config)
from configs.common.datasets.change_detection.val.whu_cd import config as val_dataset_config
config.update(val_dataset_config)

# model config
from configs.common.models.change_detection.change_mamba import model_tiny_cfg as model_cfg
config['model'] = model_cfg

# criterion config
import criteria
config['criterion'] = {'class': criteria.vision_2d.change_detection.STMambaBCDCriterion, 'args': {}}

# seeds
config['init_seed'] = 53101915
config['train_seeds'] = [63119893, 24755340, 26777559, 85907996, 67029727, 55754652, 8036857, 7601799, 11290903, 86563721, 64762802, 14395673, 51139994, 39465276, 25996136, 69864057, 98408803, 36243483, 56657755, 12553268, 8838076, 83660235, 55049471, 65136772, 8203551, 6570084, 46404347, 3624821, 47702411, 31786581, 58575373, 55928035, 15809389, 89963210, 16122616, 84465519, 60419848, 67633591, 89030843, 93773280, 56269276, 67559272, 27797023, 27894662, 14411949, 64134609, 66639569, 54685965, 27080069, 53657336, 69009301, 26097962, 11845361, 76424220, 4041836, 76467992, 64563292, 86981749, 12798508, 15532449, 42244085, 77870601, 2923277, 79362211, 12940696, 42780825, 11616980, 11738447, 22273467, 91422428, 8429963, 24937867, 56297514, 55242779, 26728842, 29945263, 15200548, 74540438, 26958548, 83161031, 48872012, 99376310, 47185164, 5787510, 36619501, 76573414, 94773639, 38978116, 93816560, 26655153, 70471570, 92886389, 92385270, 47189671, 97768769, 47079814, 89633513, 60459097, 9124887, 52821825]

# work dir
config['work_dir'] = "./logs/benchmarks/change_detection/whu_cd/ChangeMamba-Tiny_run_0"
