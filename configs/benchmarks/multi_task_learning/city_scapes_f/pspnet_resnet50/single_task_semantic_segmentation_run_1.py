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
                'labels': ['semantic_segmentation'],
                'meta_info': ['image_filepath', 'image_resolution'],
            },
        },
    }
dataset_config['criterion'] = dataset_config['criterion']['args']['criterion_configs']['semantic_segmentation']
dataset_config['metric'] = dataset_config['metric']['args']['metric_configs']['semantic_segmentation']
config.update(dataset_config)

# model config
from configs.common.models.city_scapes_f.pspnet_resnet50 import model_config_semantic_segmentation as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config as optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 56994441
config['train_seeds'] = [87493338, 26482286, 89778294, 97762444, 87236975, 23107589, 18073941, 76572813, 71949681, 26556338, 16931122, 58473660, 43153077, 67802818, 52646526, 28973925, 61432444, 78957369, 42260587, 57393491, 70445129, 93963252, 72951624, 3737643, 9114756, 39689190, 68845554, 32601857, 29420411, 64303949, 54711790, 61279249, 26924575, 36678549, 69360898, 53465622, 16633379, 30971344, 53272177, 72505183, 36732179, 32227994, 56602150, 22374132, 40496148, 81676662, 65620969, 33789946, 11287475, 4564364, 46319873, 88706767, 34892898, 14496232, 53528594, 33511120, 45977373, 56675632, 90180337, 15280540, 515485, 81244724, 68512800, 11454959, 68819960, 53810807, 14519361, 89794682, 54823943, 95103575, 28845438, 68636341, 66080899, 24309750, 76088357, 45809640, 24388010, 47305300, 61564587, 711250, 62155220, 54802700, 57283558, 54408658, 46132633, 71893905, 13074354, 97082275, 62814811, 75837296, 61143950, 66773329, 46335599, 43309244, 54751851, 55014583, 75789000, 31282549, 61416520, 70978166]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_f/pspnet_resnet50/single_task_semantic_segmentation_run_1"
