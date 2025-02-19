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
from configs.common.datasets.change_detection.train.sysu_cd import config as train_dataset_config
config.update(train_dataset_config)
from configs.common.datasets.change_detection.val.sysu_cd import config as val_dataset_config
config.update(val_dataset_config)

# model config
from configs.common.models.change_detection.ftn import model_config
config['model'] = model_config

# criterion config
import criteria
config['criterion'] = {'class': criteria.vision_2d.change_detection.FTNCriterion, 'args': {}}

# seeds
config['init_seed'] = 26248044
config['train_seeds'] = [27419391, 1901268, 69269653, 44985882, 81139791, 92710514, 36873890, 32945982, 41908451, 83833218, 97759819, 88126345, 89488079, 5886669, 47132164, 13153013, 39771977, 30651266, 72749592, 35290376, 13970044, 23995394, 73919927, 94068038, 43661280, 31603013, 75136312, 52735589, 90216393, 52028060, 67445157, 3876603, 32560328, 84742011, 4603596, 65759934, 64894472, 77684805, 20817204, 53570201, 45164422, 57277221, 82403117, 93637253, 76283019, 74575261, 23911536, 63203128, 39740426, 25410146, 31024028, 97902323, 23722585, 47716496, 58601268, 19797377, 98825754, 33116219, 54023091, 28615664, 21098601, 74079377, 86454883, 37323316, 41796386, 32005981, 80428924, 44037253, 69325079, 19654941, 95981857, 5738750, 85736538, 76457040, 63187555, 93842654, 4577150, 250288, 44283769, 97942607, 51742850, 89083151, 63896920, 70378894, 14076845, 32256699, 80927469, 38461562, 19331576, 74711238, 8890554, 94502876, 32837037, 36573006, 19424662, 31205063, 25965905, 77885593, 57668626, 5049131]

# work dir
config['work_dir'] = "./logs/benchmarks/change_detection/sysu_cd/FTN_run_1"
