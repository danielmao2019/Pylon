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

from runners import SupervisedSingleTaskTrainer
config['runner'] = SupervisedSingleTaskTrainer

# dataset config
import data
from configs.common.datasets.multi_task_learning.city_scapes_f import config as dataset_config
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
from configs.common.models.multi_task_learning.city_scapes_f.pspnet_resnet50 import model_config_instance_segmentation as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config as optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 36565127
config['train_seeds'] = [94823422, 96240388, 97917448, 54561494, 30844908, 30458690, 79705758, 45323241, 71237167, 52818073, 75849298, 45010895, 40476585, 65672135, 58881376, 80016947, 17135802, 18620957, 1451258, 29920979, 86503985, 77580779, 49045068, 7077162, 58509374, 86582731, 54347016, 58023898, 38387354, 27709391, 49823520, 22485282, 41996620, 89101269, 66117467, 29470162, 70702550, 40972, 38660702, 67477080, 75005303, 68721653, 23567758, 47714767, 21985999, 80262880, 76798134, 78032092, 42375024, 42834305, 29144673, 24542621, 9228772, 28456253, 73034161, 24159154, 31160854, 37634400, 11136910, 28274024, 11417768, 49616721, 43489284, 70665680, 62312365, 38843876, 97018109, 49132686, 99452623, 66139577, 85678298, 8042908, 56283888, 51945449, 99702138, 19771971, 90715522, 26421709, 27342307, 52870888, 34124843, 86492317, 69042552, 77450373, 78608661, 42110300, 7026379, 41192238, 97102457, 67486011, 10378988, 8470642, 1430546, 44312582, 48289243, 17693705, 9745845, 72148978, 95792269, 49093105]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/city_scapes_f/pspnet_resnet50/single_task_instance_segmentation_run_1"
