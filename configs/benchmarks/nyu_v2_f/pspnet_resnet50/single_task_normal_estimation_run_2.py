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

from runners import SupervisedSingleTaskTrainer
config['runner'] = SupervisedSingleTaskTrainer

# dataset config
import data
from configs.common.datasets.nyu_v2_f import config as dataset_config
for key in ['train_dataset', 'val_dataset', 'test_dataset']:
    dataset_config[key] = {
        'class': data.datasets.ProjectionDatasetWrapper,
        'args': {
            'dataset_config': dataset_config[key],
            'mapping': {
                'inputs': ['image'],
                'labels': ['normal_estimation'],
                'meta_info': ['image_filepath', 'image_resolution'],
            },
        },
    }
dataset_config['criterion'] = dataset_config['criterion']['args']['criterion_configs']['normal_estimation']
dataset_config['metric'] = dataset_config['metric']['args']['metric_configs']['normal_estimation']
config.update(dataset_config)

# model config
from configs.common.models.nyu_v2_f.pspnet_resnet50 import model_config_normal_estimation as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers._core_ import adam_optimizer_config
config['optimizer'] = adam_optimizer_config

# seeds
config['init_seed'] = 68928783
config['train_seeds'] = [70432212, 30042670, 41533314, 46446287, 92633780, 49538225, 50082248, 52961314, 70814084, 86984879, 34415236, 43413733, 9808802, 92972681, 63138979, 92900204, 15798448, 41498326, 56126772, 35245189, 8586613, 59035654, 17116365, 17776342, 21793639, 47384787, 67063549, 44605503, 98044304, 46423120, 29231508, 37354591, 81181933, 55293118, 88085141, 71976460, 61915539, 73400361, 89611837, 60193086, 74148899, 93169959, 9568395, 50472997, 70350673, 69463658, 52888724, 54838341, 31403727, 3059225, 89438419, 40899400, 75666388, 39088226, 56159701, 70986548, 12525833, 69637935, 60507876, 52451575, 71295872, 18941347, 36730475, 21274960, 5086782, 79920540, 22700399, 86824928, 67164718, 14405954, 62765567, 13965024, 38336618, 64586713, 4020037, 49924894, 60873720, 48781717, 97015499, 96792176, 5126274, 80859518, 22204087, 7636763, 30915265, 95290948, 9422126, 64573201, 56333067, 62199777, 45599217, 8282813, 18141840, 94798281, 21678036, 6335933, 51349219, 69996882, 23432266, 71121681]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_f/pspnet_resnet50/single_task_normal_estimation_run_2"
