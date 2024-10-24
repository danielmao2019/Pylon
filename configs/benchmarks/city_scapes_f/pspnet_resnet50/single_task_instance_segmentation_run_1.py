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
config['init_seed'] = 82981382
config['train_seeds'] = [76369101, 23428025, 87298089, 33925146, 76118661, 32828907, 28023669, 11066520, 4061367, 64545090, 27388447, 43515650, 47017591, 79087655, 85994092, 65025156, 49905900, 95018618, 16602519, 80381600, 74952572, 90096680, 88550122, 83512009, 32503117, 43824961, 61889992, 53460162, 20903577, 35678435, 22422871, 25922986, 75373385, 32627884, 31532830, 90194382, 94528837, 67084408, 34170497, 43094764, 70866423, 57080989, 89622703, 856412, 78182018, 47611903, 54019122, 21076527, 38447853, 74425690, 65638968, 58501790, 99452626, 99047607, 78882898, 61809482, 33716922, 55011799, 59350051, 20578124, 34857025, 17943788, 94469394, 84899641, 78510266, 84949095, 11365389, 64916516, 98783543, 42001246, 25658708, 58178060, 65558200, 94304218, 51311674, 98433170, 60739148, 25998580, 17860296, 29078633, 45699297, 24118133, 75401396, 52792903, 88397756, 75792807, 95023137, 37381612, 31218159, 64375999, 1575052, 37976939, 43389917, 19448160, 72545385, 50724746, 48528176, 75254936, 69486216, 25512647]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_f/pspnet_resnet50/single_task_instance_segmentation_run_1"
