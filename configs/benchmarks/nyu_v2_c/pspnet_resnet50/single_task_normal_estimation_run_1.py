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
from configs.common.datasets.nyu_v2_c import config as dataset_config
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
from configs.common.models.nyu_v2_c.pspnet_resnet50 import model_config_normal_estimation as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config as optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 4746577
config['train_seeds'] = [74752026, 91520562, 26304699, 27548217, 22960826, 58688492, 6580363, 19232083, 70675030, 71454216, 4769153, 65038682, 31388645, 25216072, 71139135, 29199424, 12853005, 80579543, 4319860, 54417293, 82258882, 15547644, 44528451, 41560164, 22324149, 62373722, 24339087, 39271859, 27723190, 91650784, 63282264, 77393970, 84796195, 95308954, 78991264, 10469872, 43846508, 73194928, 99499278, 77223391, 75317696, 74214953, 4239493, 38365945, 94610977, 85265791, 89901544, 20213215, 34067997, 57463337, 46747849, 61144600, 97356625, 79763780, 61848787, 62435600, 82980680, 65775332, 85079532, 29483164, 64826636, 2014531, 45240252, 82124614, 63034423, 56860608, 15697259, 75598445, 57152232, 73543205, 59882990, 69702022, 76806132, 70558514, 15593656, 36181640, 52112789, 73393261, 49113701, 6757544, 54848791, 35874084, 92521019, 93974739, 51874862, 4662206, 80377165, 37334317, 70352745, 73809213, 49243850, 92174664, 4189615, 82654823, 83388185, 64543648, 6256011, 24560103, 90781985, 9142753]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_c/pspnet_resnet50/single_task_normal_estimation_run_1"
