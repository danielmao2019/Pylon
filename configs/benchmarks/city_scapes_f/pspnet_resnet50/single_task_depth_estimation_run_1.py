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
                'labels': ['depth_estimation'],
                'meta_info': ['image_filepath', 'image_resolution'],
            },
        },
    }
dataset_config['criterion'] = dataset_config['criterion']['args']['criterion_configs']['depth_estimation']
dataset_config['metric'] = dataset_config['metric']['args']['metric_configs']['depth_estimation']
config.update(dataset_config)

# model config
from configs.common.models.city_scapes_f.pspnet_resnet50 import model_config_depth_estimation as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config as optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 45349134
config['train_seeds'] = [84097524, 66752679, 44998421, 46939874, 73430687, 5435307, 49476342, 66329188, 48304798, 57783598, 76691720, 93372395, 571486, 17175777, 35290993, 82441867, 18447685, 12547538, 90275434, 40767403, 91940942, 531, 76442422, 27303424, 61267951, 70502204, 66895410, 42720100, 57088876, 82770521, 78982570, 68863041, 818939, 85302072, 19009352, 24672219, 28402983, 69305956, 29383475, 7232819, 16888026, 35774872, 99694811, 30874908, 77986012, 27347049, 49918954, 56282780, 96316670, 29690901, 29304274, 35487326, 8294813, 49050572, 88796754, 68712845, 54910504, 41205643, 6325353, 98058441, 81414946, 55671816, 88505333, 63047521, 22183850, 36074683, 6210721, 3331404, 45213604, 32227095, 6988029, 30083817, 34229752, 44064306, 79915736, 40700590, 10518070, 69638362, 85397035, 36324991, 16088734, 26840334, 17488780, 56721992, 14132038, 43279973, 43742686, 15126585, 84419680, 39484000, 15728174, 94321141, 41540228, 48444511, 11227680, 1515716, 62846873, 8839875, 21867047, 85378467]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_f/pspnet_resnet50/single_task_depth_estimation_run_1"
