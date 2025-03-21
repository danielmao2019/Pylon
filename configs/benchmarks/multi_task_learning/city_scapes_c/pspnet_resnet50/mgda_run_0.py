# This file is automatically generated by `./configs/benchmarks/multi_task_learning/gen_multi_task_learning.py`.
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
from configs.common.datasets.multi_task_learning.city_scapes_c import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.city_scapes_c.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.mgda import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 86082211
config['train_seeds'] = [45551578, 73270812, 46485217, 6279614, 60133381, 56823554, 12591606, 42215090, 47565835, 85101531, 1564161, 91409773, 90678680, 22426284, 42939019, 11603701, 85782148, 64484598, 39640542, 19201527, 35459819, 50796258, 92685038, 69913233, 51462270, 62625389, 16218570, 10099150, 62320070, 50778023, 64126016, 49086634, 40075190, 78706422, 59971246, 32726583, 47423200, 37512459, 37106418, 26264159, 88630435, 47893347, 15776431, 68439236, 60415152, 16901204, 18596257, 38846605, 99102859, 27129187, 99984488, 25452999, 68745135, 37461594, 1263512, 95395495, 71727146, 70473160, 62762589, 28377434, 11570615, 94022938, 22663340, 40629185, 24244015, 51394206, 80036086, 46818377, 79859251, 19414092, 5594158, 83920549, 4989058, 82778036, 25613863, 95989624, 11498456, 43125657, 77520309, 21078138, 92083729, 81538368, 13559398, 65387669, 29671105, 69189235, 87937513, 79846584, 71143450, 98417234, 77005366, 85113949, 83386816, 83672489, 31945056, 69241970, 29634094, 44936743, 99437875, 15591712]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/city_scapes_c/pspnet_resnet50/mgda_run_0"
