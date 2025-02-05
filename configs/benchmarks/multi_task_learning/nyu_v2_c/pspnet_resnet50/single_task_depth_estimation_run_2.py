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
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config as optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 41237601
config['train_seeds'] = [71701621, 69278583, 27818316, 85350342, 13945629, 51068419, 53539759, 85214715, 24056895, 78748321, 95690946, 65403649, 57732970, 66326734, 20613294, 872367, 43968043, 71453118, 47848970, 44269087, 43000551, 65173868, 86185686, 90012667, 95626552, 5062114, 29487208, 46953323, 39962175, 83822517, 33390432, 94961215, 9033410, 81060988, 49867051, 60652921, 52822302, 86540663, 22208905, 45207362, 26593812, 84974180, 23695672, 88859348, 53010567, 4422237, 84377122, 14220336, 93756086, 58768110, 3445513, 48373175, 19821680, 16403288, 50820909, 44529349, 35071551, 99256180, 60238929, 84744730, 85901965, 10220847, 67276703, 11671198, 79949740, 85567164, 82027456, 13209957, 77795017, 33121399, 60196794, 42475525, 66894865, 54820920, 34808392, 59631126, 45674160, 53611079, 38332340, 97734457, 45623022, 15636951, 79847567, 11242952, 12681558, 93708838, 40830855, 62915377, 23138815, 32052825, 81453177, 14906043, 55644311, 22271736, 99593621, 36528476, 15212458, 98617698, 63966681, 37512288]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_c/pspnet_resnet50/single_task_depth_estimation_run_2"
