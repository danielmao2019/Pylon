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
from configs.common.datasets.nyu_v2_c import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.nyu_v2_c.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.pcgrad import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 3940221
config['train_seeds'] = [67574711, 28418867, 51442191, 9021576, 26521366, 4572784, 27644572, 78688436, 14261442, 5964735, 89597846, 18713982, 88914663, 95681094, 92751154, 14990304, 85324790, 81608620, 2336052, 48827257, 53048218, 34265938, 67325653, 19361713, 47349143, 9140844, 44008010, 95603001, 7046003, 23508525, 57465722, 83409217, 44500924, 25771615, 35223131, 95263433, 44539236, 87574593, 60847737, 66337274, 65246382, 59886256, 12169563, 9000758, 73711268, 92885059, 83454387, 43122783, 52476471, 76041383, 55654737, 12110182, 12432607, 93308252, 80047546, 2059845, 55835923, 72223097, 8608746, 16393661, 60806011, 31836599, 45436984, 34033378, 35433699, 19203865, 66812328, 13035063, 5001859, 24624090, 51174734, 74916468, 39121741, 10342269, 3946099, 77158176, 27735768, 54387553, 22144955, 3055053, 85949699, 76316400, 11377888, 32262938, 89022107, 83977339, 70799363, 32894742, 64232338, 71473320, 43238428, 66754394, 24246619, 53654307, 44194734, 26692552, 81121652, 73873024, 2953453, 88743878]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_c/pspnet_resnet50/pcgrad_run_2"
