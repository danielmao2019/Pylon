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
from configs.common.datasets.city_scapes_f import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.city_scapes_f.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.imtl import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 16359800
config['train_seeds'] = [14161173, 72840856, 82448359, 51930602, 35232943, 89381927, 28767812, 65543284, 36029397, 68059643, 35351783, 13026711, 52494424, 85910363, 12934876, 94986704, 3597846, 1939328, 45373854, 93616490, 25636250, 9735520, 96217651, 64786077, 84559521, 97647540, 470104, 34476878, 39616892, 84918985, 98328732, 8463370, 81152488, 50805470, 32568641, 66578946, 99848084, 2690214, 97564736, 4396845, 12726871, 7608405, 74576624, 23105908, 66680264, 83408286, 18493668, 22582851, 47632396, 46827076, 76342001, 18153435, 29815679, 9117720, 31511019, 8604117, 73156179, 29502968, 74718799, 76976315, 14199460, 80861499, 60384430, 1356122, 7974139, 33654867, 96733075, 36884387, 91311955, 26772044, 79066752, 97367145, 56144150, 18319648, 9550320, 42142041, 85001313, 68333442, 91346357, 562743, 84733980, 49954711, 93991911, 74043101, 51030764, 82916169, 19233325, 70294260, 80894216, 16443418, 20173773, 9528105, 43945655, 31010163, 49388328, 7815010, 96848283, 45179063, 30796739, 65636874]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_f/pspnet_resnet50/imtl_run_0"
