# This file is automatically generated by `./configs/benchmarks/change_detection/gen_ppsl.py`.
# Please do not attempt to modify manually.
import torch
import schedulers


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
    'val_dataset': None,
    'val_dataloader': None,
    'test_dataset': None,
    'test_dataloader': None,
    'criterion': None,
    'metric': None,
    # model config
    'model': None,
    # optimizer config
    'optimizer': None,
    'scheduler': {
        'class': torch.optim.lr_scheduler.LambdaLR,
        'args': {
            'lr_lambda': {
                'class': schedulers.WarmupLambda,
                'args': {},
            },
        },
    },
}

from runners import SupervisedSingleTaskTrainer
config['runner'] = SupervisedSingleTaskTrainer

# dataset config
from configs.common.datasets.change_detection.ppsl_whu_bd_levir_cd import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.change_detection.ppsl_model import model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config as optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 40090936
config['train_seeds'] = [65237392, 21373085, 45868611, 23645924, 99076260, 79371121, 16543517, 43478082, 42881130, 720772, 72342778, 24715755, 32852015, 67192881, 90294475, 16543130, 32863392, 39635785, 84173327, 50609609, 9353685, 15062887, 27211923, 66501996, 52463918, 92242419, 85219445, 3887453, 57540907, 36906058, 73720355, 92241560, 19968573, 51987349, 11301451, 32699401, 88446762, 10979192, 76176426, 73456850, 96493090, 59036295, 86656884, 85614996, 20010810, 26363297, 39158723, 1450972, 49768689, 90573920, 32415703, 32808854, 89634597, 67262002, 97294588, 46198571, 30290627, 54580386, 92099478, 63414260, 21472169, 69835752, 40396151, 98580929, 90895191, 53956646, 93458331, 82371982, 94177127, 81426659, 84840181, 33932607, 29180473, 47253754, 61964979, 46466612, 24362007, 65738283, 65628045, 47150148, 92622532, 98628523, 87751551, 76109083, 59029171, 20384348, 97559915, 8414332, 20718588, 98310103, 80067455, 49293771, 89182192, 25767844, 14434408, 63096383, 39581338, 53419791, 62416287, 634402]

# work dir
config['work_dir'] = "./logs/benchmarks/change_detection/ppsl/ppsl_run_2"
