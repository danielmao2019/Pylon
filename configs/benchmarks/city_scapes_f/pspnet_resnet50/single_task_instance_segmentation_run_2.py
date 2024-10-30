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
                'labels': ['instance_segmentation'],
                'meta_info': ['image_filepath', 'image_resolution'],
            },
        },
    }
dataset_config['criterion'] = dataset_config['criterion']['args']['criterion_configs']['instance_segmentation']
dataset_config['metric'] = dataset_config['metric']['args']['metric_configs']['instance_segmentation']
config.update(dataset_config)

# model config
from configs.common.models.city_scapes_f.pspnet_resnet50 import model_config_instance_segmentation as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers._core_ import adam_optimizer_config
config['optimizer'] = adam_optimizer_config

# seeds
config['init_seed'] = 38540530
config['train_seeds'] = [76005811, 53180609, 5928161, 4516880, 22772239, 20292645, 47673263, 17009257, 4086591, 49495964, 30471490, 22201486, 83471297, 84932740, 25805354, 14684667, 56067067, 44986015, 19482870, 35057004, 16917569, 26469867, 90389008, 31140781, 38255242, 6724213, 17473523, 84105869, 62035596, 990044, 9408281, 95547877, 19390739, 23677134, 16165209, 57756748, 9543442, 40429175, 22681646, 28010940, 48307009, 84052446, 79829514, 58390166, 50141264, 83181848, 50621121, 75452349, 471418, 58020767, 7533908, 91788759, 8828503, 46561554, 66113721, 2884329, 98960222, 98155212, 74683543, 13323616, 89174185, 88322387, 54524875, 93028241, 13415883, 7816531, 23406402, 46898908, 39821032, 92265476, 80290540, 21717591, 67675982, 57159861, 22825434, 20622946, 60545068, 14358377, 86587150, 13978808, 17699965, 30831089, 22910909, 14595228, 10225167, 97525829, 14470703, 69368618, 20619997, 64931158, 59849651, 83155460, 16864724, 66474718, 24728216, 93695935, 47002952, 58054475, 75690396, 8395514]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_f/pspnet_resnet50/single_task_instance_segmentation_run_2"