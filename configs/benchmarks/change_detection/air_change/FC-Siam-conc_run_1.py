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
from configs.common.datasets.change_detection.train.air_change import config as train_dataset_config
config.update(train_dataset_config)
from configs.common.datasets.change_detection.val.air_change import config as val_dataset_config
config.update(val_dataset_config)

# model config
from configs.common.models.change_detection.fc_siam import model_config
config['model'] = model_config
config['model']['args']['arch'] = "FC-Siam-conc"
config['model']['args']['in_channels'] = 3

# seeds
config['init_seed'] = 57708735
config['train_seeds'] = [36147001, 39513331, 88254492, 40058017, 53075999, 94887770, 56186111, 31881670, 82163575, 37425902, 54662483, 37086735, 42583559, 18233668, 83871253, 30090677, 93167798, 9161731, 64993960, 47157831, 54684818, 9918468, 57237747, 58664669, 17732587, 56139180, 45797660, 36844969, 2994390, 25237880, 34057321, 49341618, 94997101, 35022470, 45336764, 28263073, 46881425, 19982575, 52255256, 75624401, 81842993, 18254823, 93158223, 618448, 91351595, 30432965, 82735881, 25199054, 17064440, 45793522, 79642177, 61451969, 58628213, 21279909, 98108718, 28434263, 31710554, 78599447, 3742691, 97822881, 1209179, 67398422, 10300509, 52178208, 54577056, 76504867, 11463966, 45186115, 86420076, 64498058, 41368912, 91108050, 50164281, 1075751, 97165893, 51678407, 6712872, 64094736, 9874127, 12725994, 44646232, 45530288, 49724943, 55136872, 71673939, 84229408, 20520976, 39304619, 2643570, 90311791, 25245689, 15484088, 11401330, 36219127, 34962866, 14427401, 75470494, 42931787, 19960457, 35478957]

# work dir
config['work_dir'] = "./logs/benchmarks/change_detection/air_change/FC-Siam-conc_run_1"
