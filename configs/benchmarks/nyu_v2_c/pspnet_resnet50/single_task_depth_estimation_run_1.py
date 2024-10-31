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
                'labels': ['depth_estimation'],
                'meta_info': ['image_filepath', 'image_resolution'],
            },
        },
    }
dataset_config['criterion'] = dataset_config['criterion']['args']['criterion_configs']['depth_estimation']
dataset_config['metric'] = dataset_config['metric']['args']['metric_configs']['depth_estimation']
config.update(dataset_config)

# model config
from configs.common.models.nyu_v2_c.pspnet_resnet50 import model_config_depth_estimation as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers._core_ import adam_optimizer_config
config['optimizer'] = adam_optimizer_config

# seeds
config['init_seed'] = 69580935
config['train_seeds'] = [95727796, 61829015, 15194620, 65149946, 84883509, 43865392, 55247492, 44935228, 31080561, 12996085, 63397117, 79787159, 36522743, 41158458, 32456453, 79272200, 74030392, 20881005, 76113113, 85553750, 93437071, 83324456, 63098023, 29121967, 51937558, 49327336, 20916356, 5710374, 16884518, 19148220, 13322070, 55634033, 97463610, 40784733, 3309262, 89176679, 18353416, 6235183, 97209329, 83427301, 51675329, 15750955, 1005814, 66132320, 39024037, 18768070, 74328306, 4898350, 25450398, 53728513, 33655870, 41534592, 63382825, 87181274, 69291165, 58664052, 748808, 85962512, 65875286, 11988196, 2931049, 92438460, 54531391, 73041323, 3025518, 11458757, 7611788, 67619078, 772261, 9621391, 11821741, 81073293, 77279003, 48498091, 97754104, 69010715, 69087409, 50073587, 82625297, 13488081, 93155431, 24130724, 12651835, 35925425, 21233216, 28658631, 54200000, 40871974, 39958353, 30656945, 90293567, 70835778, 78022271, 96556285, 29820222, 89161188, 83428121, 68566544, 50729914, 80972485]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_c/pspnet_resnet50/single_task_depth_estimation_run_1"
