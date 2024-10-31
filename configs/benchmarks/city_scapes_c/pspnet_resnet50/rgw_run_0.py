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
config['init_seed'] = 49169483
config['train_seeds'] = [37668790, 37979510, 32565300, 97501106, 81922087, 87853189, 9279952, 96252838, 13065646, 69484416, 30579376, 59582929, 93776054, 36425883, 67743345, 60905094, 24952331, 42624946, 46905776, 41853347, 91552092, 13079409, 48482222, 17536715, 2068318, 66699119, 94835010, 60779825, 22775072, 14048171, 93384311, 18543050, 75292842, 44755134, 84500904, 50872725, 23798139, 90383248, 57038009, 17504816, 14479050, 71961448, 82306774, 61973568, 21840146, 55541173, 4878503, 80081142, 60331803, 31903797, 49085423, 32315548, 94059132, 46938885, 1956034, 63415799, 87419931, 14499061, 71937298, 86451531, 39312089, 10524127, 24382156, 90616769, 23622298, 58153147, 50611359, 81951255, 75660323, 74108186, 46330261, 90204713, 1103108, 83891716, 56270106, 16261057, 2368675, 67549324, 2415537, 15767070, 4401008, 23428631, 60586640, 88277684, 32022590, 8243600, 8200315, 99909402, 7661114, 72152508, 89808454, 22924079, 70712628, 65316878, 58261269, 16834137, 13796624, 84789069, 86512664, 87678899]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_c/pspnet_resnet50/rgw_run_0"
