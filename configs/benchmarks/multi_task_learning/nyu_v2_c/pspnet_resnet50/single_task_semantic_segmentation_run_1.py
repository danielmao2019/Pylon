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
                'labels': ['semantic_segmentation'],
                'meta_info': ['image_filepath', 'image_resolution'],
            },
        },
    }
dataset_config['criterion'] = dataset_config['criterion']['args']['criterion_configs']['semantic_segmentation']
dataset_config['metric'] = dataset_config['metric']['args']['metric_configs']['semantic_segmentation']
config.update(dataset_config)

# model config
from configs.common.models.nyu_v2_c.pspnet_resnet50 import model_config_semantic_segmentation as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config as optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 78160941
config['train_seeds'] = [41405384, 37317910, 99048474, 70457425, 48964295, 64824125, 31142942, 56667518, 4532340, 63047027, 50484063, 82394220, 45164364, 15775646, 83934006, 13170731, 33314914, 75536128, 97857772, 8771164, 3122743, 40504355, 61238971, 39898247, 44941769, 76337391, 75445224, 94353609, 63318915, 18198193, 87303531, 71184979, 64918061, 76057004, 66351557, 23725450, 75772432, 9867753, 22089563, 2789827, 96884294, 93769715, 84373658, 46051569, 94672593, 1527224, 505732, 69196996, 91824197, 21856128, 73912568, 78999045, 60801770, 70093578, 14612318, 57223113, 47322134, 71719687, 53059619, 24902301, 97550911, 53456298, 7279240, 236829, 19388093, 48934108, 45700503, 22098899, 63304916, 79608733, 2357623, 39118301, 50389121, 99273804, 76789863, 16911564, 99322076, 17032066, 81549709, 61039977, 3659220, 5563277, 42786776, 33988157, 8952790, 74854072, 58576013, 63533414, 51422267, 74898974, 25152083, 36597170, 83458094, 35986865, 96247756, 8648438, 97330681, 48482943, 10321089, 35696824]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_c/pspnet_resnet50/single_task_semantic_segmentation_run_1"
