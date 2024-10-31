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
from configs.common.datasets.city_scapes_f import config as dataset_config
for key in ['train_dataset', 'val_dataset', 'test_dataset']:
    dataset_config[key] = {
        'class': data.datasets.ProjectionDatasetWrapper,
        'args': {
            'dataset_config': dataset_config[key],
            'mapping': {
                'inputs': ['image'],
                'labels': ['instance_segmentation'],
                'meta_info': ['image_filepath', 'image_resolution'],
            },
        },
    }
dataset_config['criterion'] = dataset_config['criterion']['args']['criterion_configs']['instance_segmentation']
dataset_config['metric'] = dataset_config['metric']['args']['metric_configs']['instance_segmentation']
config.update(dataset_config)

# model config
from configs.common.models.city_scapes_f.pspnet_resnet50 import model_config_instance_segmentation as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers._core_ import adam_optimizer_config
config['optimizer'] = adam_optimizer_config

# seeds
config['init_seed'] = 89316380
config['train_seeds'] = [63398132, 19204284, 36035428, 97218673, 70347581, 62249236, 29101256, 13152891, 23983364, 65194923, 76065854, 64613136, 31136553, 20830153, 25199537, 69293393, 39283955, 58129171, 64908861, 13233869, 61598888, 73438944, 47927929, 800985, 87134372, 79107734, 50376280, 38109035, 30853643, 39761417, 92526433, 37336407, 73332702, 44707040, 53904019, 13469414, 83027377, 94836131, 18600755, 31872616, 91082750, 17252053, 25172018, 87624528, 39680778, 68310939, 21767592, 15893741, 97817264, 91871490, 38277730, 18617980, 81220507, 96575842, 78197201, 92200098, 87977122, 14051554, 13710662, 54041865, 28602163, 89921804, 60625382, 53139238, 35147639, 45227265, 32409679, 77233037, 77819025, 73861271, 33862767, 35258614, 20026601, 59763659, 88380027, 21095025, 44118037, 11604011, 74155696, 94966540, 26103292, 25932708, 55638042, 45018462, 10832702, 95808980, 77343977, 52112487, 83513849, 55412656, 22877831, 20497497, 99947493, 63715135, 87860511, 46340251, 60628193, 94833967, 1529302, 77356667]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_f/pspnet_resnet50/single_task_instance_segmentation_run_2"
