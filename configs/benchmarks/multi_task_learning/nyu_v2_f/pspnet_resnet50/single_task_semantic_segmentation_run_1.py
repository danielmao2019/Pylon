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
                'labels': ['semantic_segmentation'],
                'meta_info': ['image_filepath', 'image_resolution'],
            },
        },
    }
dataset_config['criterion'] = dataset_config['criterion']['args']['criterion_configs']['semantic_segmentation']
dataset_config['metric'] = dataset_config['metric']['args']['metric_configs']['semantic_segmentation']
config.update(dataset_config)

# model config
from configs.common.models.nyu_v2_f.pspnet_resnet50 import model_config_semantic_segmentation as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config as optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 71187827
config['train_seeds'] = [98178661, 82636059, 19297698, 15967645, 98877145, 25728037, 98125620, 78689051, 62787554, 5466139, 83651965, 34552547, 77663224, 47225507, 55868033, 58491786, 52701095, 4873580, 55348361, 1743873, 64484807, 84484848, 43157015, 65328559, 99386919, 20889146, 26359116, 6180452, 42061281, 65498675, 38778234, 50021890, 55847202, 27552335, 61864984, 87309241, 72197719, 52026000, 95031989, 26212209, 12846903, 70664885, 52630657, 99908160, 39608360, 11940514, 44888691, 63705495, 73194576, 28328004, 56377421, 41328630, 67713320, 2876221, 60308014, 84786645, 57121618, 46200943, 39312880, 87942201, 37913293, 60994039, 57645058, 39549423, 8105893, 33242414, 51256046, 70323177, 84145814, 67140687, 94169342, 89606875, 31716602, 67321930, 62185473, 78082248, 71013966, 51832264, 63879642, 62725937, 34824176, 64744500, 94942173, 24938530, 17746861, 35018462, 20908810, 63147261, 93934033, 8162882, 2404866, 1891550, 83832555, 19616760, 21187769, 28774811, 10488501, 89753290, 28331954, 58553000]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_f/pspnet_resnet50/single_task_semantic_segmentation_run_1"
