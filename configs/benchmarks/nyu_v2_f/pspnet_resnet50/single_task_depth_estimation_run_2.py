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
                'labels': ['depth_estimation'],
                'meta_info': ['image_filepath', 'image_resolution'],
            },
        },
    }
dataset_config['criterion'] = dataset_config['criterion']['args']['criterion_configs']['depth_estimation']
dataset_config['metric'] = dataset_config['metric']['args']['metric_configs']['depth_estimation']
config.update(dataset_config)

# model config
from configs.common.models.nyu_v2_f.pspnet_resnet50 import model_config_depth_estimation as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config as optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 96759876
config['train_seeds'] = [94777389, 11828245, 21518459, 74653758, 11321162, 64441541, 96432953, 57224529, 46867017, 14680589, 15169413, 915307, 28075284, 66560799, 71910009, 30515392, 39786036, 40638642, 93710338, 8213942, 91207540, 66335797, 62670309, 79208072, 48604211, 69006708, 23911828, 96178242, 54172048, 45911521, 42810036, 24369965, 3466138, 69689321, 27172378, 58675201, 69880937, 20494689, 11858232, 54060902, 89777093, 97416017, 52926289, 22036909, 18692484, 87405724, 76461512, 57810784, 25766042, 89200063, 95823205, 31507843, 73949910, 23038879, 75822941, 46868762, 32388599, 3100424, 26025747, 62836137, 96727450, 59261525, 22644434, 60980833, 96188823, 32563541, 36041921, 52970838, 27962666, 23655210, 41885600, 81443925, 78128948, 14632310, 84687222, 59549477, 25994715, 1642225, 30425533, 22063720, 3254090, 90491257, 56491680, 54805375, 14962089, 25020491, 17220510, 61653612, 13188403, 49351080, 79300392, 27338824, 42394102, 72632172, 43567289, 65018368, 61664102, 66023814, 78382297, 26843981]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_f/pspnet_resnet50/single_task_depth_estimation_run_2"
