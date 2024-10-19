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
from configs.common.optimizers._core_ import adam_optimizer_config
config['optimizer'] = adam_optimizer_config

# seeds
config['init_seed'] = 5992776
config['train_seeds'] = [19106535, 63971045, 51408179, 59088661, 38461466, 47920126, 28550031, 99163063, 18550853, 5835153, 68588850, 34185382, 77898991, 24963403, 86608213, 10859438, 11011288, 22201816, 4565988, 50223786, 88685963, 75034142, 82235328, 36824453, 15636121, 75845261, 85484060, 16628575, 1065830, 65296203, 32387726, 35744927, 57536525, 79322235, 57030760, 5537021, 51539605, 68354150, 46067100, 17685447, 39659759, 92007076, 30900211, 22550296, 88617410, 6605973, 24448494, 26388844, 90155744, 95955408, 27051706, 21353917, 5089282, 12672766, 47852659, 89130039, 59093915, 79523085, 96334395, 5923398, 66355606, 55178162, 35340088, 84602422, 61332955, 64759672, 84683375, 35914408, 83089043, 44995420, 33130535, 87028037, 23981077, 34297954, 56624556, 62962710, 14974409, 94384688, 18238184, 41603878, 54236899, 97002029, 26932172, 46897842, 88415853, 98578864, 21383882, 25698481, 68274734, 82205288, 98325072, 18846795, 45681250, 68653831, 23460082, 46780061, 4560082, 12921657, 91901581, 99348165]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_f/pspnet_resnet50/single_task_depth_estimation_run_1"
