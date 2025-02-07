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
config['init_seed'] = 721706
config['train_seeds'] = [97190565, 37839619, 72783241, 14452738, 88946957, 12642462, 25007959, 19411220, 7180454, 85811380, 12777773, 77215714, 2196307, 59482992, 42111792, 67654743, 27120342, 17669299, 19815221, 14439194, 8718643, 51235403, 65636337, 55948959, 49355797, 33691763, 42186224, 77372365, 13924011, 82177558, 65421353, 6704185, 30617250, 28018020, 11820162, 31446434, 34908710, 16831040, 89161262, 33650618, 80758250, 33984180, 55433688, 74168035, 21921181, 32152281, 85056310, 34640911, 68823397, 28876895, 10033977, 6279513, 48588704, 7150559, 2128324, 88387895, 69603913, 91360125, 33358273, 36966934, 40733940, 56935656, 19071256, 5333506, 59170535, 54411135, 88017599, 47314099, 83203667, 66263089, 95344339, 35474680, 31124027, 99042769, 60466024, 37698980, 25865650, 53525534, 81997557, 56487483, 77008927, 38710547, 53032090, 17839520, 33936021, 25853454, 47839461, 65010720, 43463608, 6071066, 62123900, 56814820, 97760070, 86481759, 15431590, 16494036, 62290798, 9159162, 6484705, 66805122]

# work dir
config['work_dir'] = "./logs/benchmarks/change_detection/ppsl/whu_bd_run_0"
