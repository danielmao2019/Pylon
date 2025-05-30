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
from configs.common.datasets.multi_task_learning.city_scapes_c import config as dataset_config
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
from configs.common.models.multi_task_learning.city_scapes_c.pspnet_resnet50 import model_config_instance_segmentation as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config as optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 73573503
config['train_seeds'] = [73771756, 75686571, 92804956, 24072176, 20619668, 57733686, 13785522, 85549363, 22541604, 2297384, 10155865, 16558050, 60208717, 22374685, 76863006, 41342475, 94554300, 73894279, 37426553, 28513495, 25706220, 89057927, 93751240, 20731666, 34884030, 5173811, 54570442, 95066098, 4047442, 79411945, 26810769, 75925409, 56574927, 39311321, 54741423, 45703802, 8788612, 58406879, 48098984, 60087323, 31331720, 36873416, 60702797, 43075055, 5340300, 82061858, 84756757, 28102784, 9175949, 96871059, 86764493, 5687905, 53371368, 3656888, 84963746, 55983603, 74272301, 75962244, 36156316, 89727567, 61660599, 7949150, 63129201, 52893311, 87797220, 78751025, 49546933, 882524, 81482534, 30865232, 83605111, 86855366, 46433477, 66266231, 32789293, 82737431, 92981731, 29651412, 86345781, 56016464, 60542663, 26627551, 52243648, 52837295, 15418509, 81676080, 68239369, 88711079, 55212958, 45161665, 81863281, 40534722, 65662993, 81945791, 35508299, 99915588, 61262346, 94029661, 18377334, 37355963]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/city_scapes_c/pspnet_resnet50/single_task_instance_segmentation_run_0"
