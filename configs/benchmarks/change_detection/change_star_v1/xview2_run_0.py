# This file is automatically generated by `./configs/benchmarks/change_detection/gen_change_star_v1.py`.
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

from runners import MultiValDatasetTrainer
config['runner'] = MultiValDatasetTrainer

# dataset config
from configs.common.datasets.change_detection.train.change_star_v1_xview2 import config as train_dataset_config
config.update(train_dataset_config)
from configs.common.datasets.change_detection.val.change_star_v1_xview2 import config as val_dataset_config
config.update(val_dataset_config)

# model config
from configs.common.models.change_detection.change_star import model_config
config['model'] = model_config

# seeds
config['init_seed'] = 17291621
config['train_seeds'] = [19473177, 39603551, 36148264, 59832060, 52296987, 13843578, 75847658, 87154507, 87686571, 39385439, 50958983, 73646782, 2178262, 996112, 2630292, 18446546, 68339579, 93668426, 13582705, 48318822, 64763171, 9192604, 13519661, 98751654, 89127579, 83607831, 35455357, 43213216, 89756490, 24030823, 71024538, 10668135, 44936636, 99056563, 7692802, 12983881, 56897038, 90584833, 17339299, 30514408, 12584526, 65074827, 67255050, 21807937, 82762580, 73593842, 33719340, 10099226, 3587338, 94127690, 85010913, 29443998, 2422275, 74704129, 84038805, 3470574, 30183944, 6104399, 6768878, 80995897, 39681164, 36369884, 22837968, 41221911, 2656436, 69301166, 23010159, 22135104, 69255282, 3138886, 48002692, 46250409, 38778211, 23616908, 76809755, 13747652, 21751041, 22549273, 48160486, 63624647, 77249011, 48223645, 60810552, 3113482, 3943539, 13715219, 6589685, 56184372, 18310812, 14754903, 82575662, 84885278, 53432107, 85564248, 17779029, 70691159, 93631870, 23500929, 92462795, 53675284]

# work dir
config['work_dir'] = "./logs/benchmarks/change_detection/change_star_v1/xview2_run_0"
