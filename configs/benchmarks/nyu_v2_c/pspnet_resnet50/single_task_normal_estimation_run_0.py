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
from configs.common.optimizers._core_ import adam_optimizer_config
config['optimizer'] = adam_optimizer_config

# seeds
config['init_seed'] = 58921243
config['train_seeds'] = [19458803, 46997618, 94125505, 52183620, 76257394, 57370260, 76302988, 39408924, 65364024, 50435594, 22929763, 21906226, 69213823, 38701042, 85703472, 33623716, 25674808, 70538729, 16069068, 61885778, 41610452, 32717158, 45155680, 45292657, 14008735, 67705253, 27090684, 89170732, 72224100, 3239779, 50149269, 92189502, 92406070, 17553208, 75537992, 22523532, 72207261, 98025205, 25452251, 58001483, 2481712, 15736770, 98941259, 59214284, 59984622, 43916774, 57799554, 8385324, 25629843, 99350161, 14241234, 89256981, 79802169, 13405160, 75878144, 23606655, 11538431, 26400146, 12222931, 32321508, 60235523, 97952180, 25663738, 63324715, 92233413, 18904781, 29070253, 13027183, 36789968, 93472055, 13504125, 46292797, 16449848, 43619537, 88665637, 6919157, 57661883, 82435391, 82223423, 9908532, 13175341, 82699896, 58054806, 80992721, 87242286, 46505077, 7337831, 56604148, 80715076, 85655881, 51093818, 77141547, 17658313, 93059959, 87266070, 31265816, 44434167, 9131155, 33917379, 47355307]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_c/pspnet_resnet50/single_task_normal_estimation_run_0"
