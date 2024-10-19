# This file is automatically generated by `./configs/benchmarks/multi_task_learning.py`.
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

from runners import SupervisedMultiTaskTrainer
config['runner'] = SupervisedMultiTaskTrainer

# dataset config
from configs.common.datasets.city_scapes_c import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.city_scapes_c.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.pcgrad import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 38631841
config['train_seeds'] = [48060238, 89614409, 94449596, 13273400, 54683609, 8176076, 85883000, 75495864, 83989303, 90880205, 14866379, 58168870, 63675647, 86990285, 13135392, 10989218, 82994354, 70574009, 55815829, 3292531, 10228924, 53980156, 96927971, 18001671, 55780448, 62601836, 69537917, 76537049, 60315089, 43041315, 16076060, 16806918, 20851426, 87300468, 35465790, 43879667, 10898265, 52102468, 93784063, 73097363, 95451665, 44818298, 20989199, 11154953, 71608332, 9162806, 84438571, 49601502, 26667691, 80620392, 88434613, 29141709, 73484744, 54719852, 78064114, 10285507, 46060974, 72284026, 38150818, 64539849, 92144673, 15808757, 30320951, 53525995, 5980714, 73592217, 94652487, 50868932, 13927595, 916909, 7573013, 35736794, 22501807, 2087821, 43991218, 49904196, 45399328, 57442028, 71716833, 20772495, 74867459, 61770258, 19111420, 10839543, 16377423, 11491047, 43030050, 89887190, 82903730, 2749765, 68983626, 5906272, 36910897, 16308694, 8722227, 72994385, 47631484, 1646898, 20905447, 42506031]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_c/pspnet_resnet50/pcgrad_run_1"
