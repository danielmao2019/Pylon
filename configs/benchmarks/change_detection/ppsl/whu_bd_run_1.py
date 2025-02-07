# This file is automatically generated by `./configs/benchmarks/change_detection/gen_ppsl.py`.
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
from configs.common.datasets.change_detection.train.ppsl_whu_bd import config as train_dataset_config
config.update(train_dataset_config)
from configs.common.datasets.change_detection.val.all_bi_temporal import config as val_dataset_config
config.update(val_dataset_config)

# model config
from configs.common.models.change_detection.ppsl_model import model_config
config['model'] = model_config

# seeds
config['init_seed'] = 80411010
config['train_seeds'] = [64015400, 46691589, 68923286, 6316353, 17078908, 3540830, 58780952, 81627013, 86947636, 88328699, 46708956, 60670073, 33627789, 73720744, 30764330, 15790292, 73532571, 37762574, 51210014, 43267290, 97044367, 26139981, 91007892, 36063756, 18672195, 69160089, 11999516, 93462491, 22383168, 41599528, 92887536, 99994706, 12634029, 94373604, 31167550, 29234335, 6884007, 70748235, 74700189, 9822322, 25337416, 28718863, 22076930, 76733673, 85607954, 60876173, 2463345, 23094442, 1617507, 97245237, 760973, 55974883, 45470265, 6146497, 29981613, 76355796, 38121655, 53802873, 57656242, 31373296, 44764231, 8241813, 11385770, 37190540, 69107083, 37378130, 99505241, 51067525, 53619386, 71906407, 92877224, 70918048, 30892064, 33750464, 71329244, 71468170, 34491425, 9754322, 34804326, 93702220, 85324999, 51465369, 97297629, 32104632, 73391235, 71784552, 71104069, 17316093, 29757569, 73034840, 47495718, 95492108, 60217401, 74522402, 21457495, 84481208, 28951102, 15467471, 40546436, 95497087]

# work dir
config['work_dir'] = "./logs/benchmarks/change_detection/ppsl/whu_bd_run_1"
