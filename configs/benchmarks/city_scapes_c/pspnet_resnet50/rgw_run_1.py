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
from configs.common.optimizers.rgw import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 93029211
config['train_seeds'] = [29986524, 82349361, 90903918, 72152930, 11533895, 73822049, 56953477, 83356678, 99616976, 79480649, 55257718, 81691270, 756341, 74375199, 97304049, 868625, 31825132, 13557788, 68154562, 91753959, 82457938, 18836029, 83200602, 8686609, 66212222, 34258253, 36558669, 46347429, 32065630, 90304739, 29944963, 57049450, 71764288, 15316358, 32493937, 93279945, 12146091, 60663597, 64782577, 50316438, 86701114, 16480148, 2801299, 75553767, 58735207, 72041689, 55748510, 40934020, 16308041, 22007447, 16866083, 36540478, 96312080, 69041373, 28669115, 9583864, 87615926, 42099597, 76382965, 48467115, 62900110, 20345472, 24106991, 83090677, 19195367, 2433482, 70341472, 31529351, 12622235, 71646818, 76324242, 12971356, 87812394, 14961497, 60054849, 4099127, 4209725, 62915094, 78362356, 87027897, 50848534, 83371265, 29489343, 44500799, 31128380, 60937350, 99246365, 63541840, 47703384, 75311992, 4340663, 22230898, 39113734, 50114946, 38155196, 8206360, 49154362, 55506514, 21074591, 82536178]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_c/pspnet_resnet50/rgw_run_1"