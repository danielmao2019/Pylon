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
    'val_seeds': None,
    'test_seed': None,
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
                'class': schedulers.lr_lambdas.WarmupLambda,
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
                'labels': ['depth_estimation'],
                'meta_info': ['image_filepath', 'image_resolution'],
            },
        },
    }
dataset_config['criterion'] = dataset_config['criterion']['args']['criterion_configs']['depth_estimation']
dataset_config['metric'] = dataset_config['metric']['args']['metric_configs']['depth_estimation']
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.city_scapes_c.pspnet_resnet50 import model_config_depth_estimation as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config as optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 48129396
config['train_seeds'] = [36241692, 92621841, 25106358, 71288564, 68345978, 27754113, 10848317, 38765098, 38091781, 53270373, 69731666, 95566976, 55801290, 92036837, 6439420, 58124961, 32614670, 46194825, 38205090, 21376589, 25727829, 35218497, 94575030, 50428115, 98639504, 59360960, 76636359, 80211305, 6792574, 41274258, 59357045, 25266784, 19727555, 99671193, 68723428, 18640431, 40665651, 15166349, 52453383, 47126937, 56443706, 34982052, 66545596, 15088840, 34740408, 91017213, 86549398, 22449072, 28674327, 94240073, 37528250, 90696003, 2296093, 33262613, 86072489, 44983681, 64936424, 78218511, 49479373, 81708869, 12303164, 88645627, 13350987, 26608144, 84729803, 34965891, 30701324, 40915937, 20271631, 32793378, 45535182, 33802938, 96993864, 70089224, 17638819, 29566189, 7972621, 95900350, 83046397, 61782780, 5644630, 91693236, 52475396, 76472822, 49598761, 26538273, 80949709, 60126194, 72209364, 68617530, 3822435, 28518080, 73289393, 74713980, 52610826, 21510735, 83779236, 49244600, 36822010, 10080590]
config['val_seeds'] = [90269037, 1951268, 25920429, 18904376, 40656471, 24995676, 61263338, 67049059, 30361327, 10999477, 53940969, 80778561, 35538022, 79937705, 27837450, 74757891, 15791821, 33762294, 68660273, 44182630, 56680658, 60112993, 28675665, 2547928, 13236348, 11933370, 19030928, 73366933, 21256089, 85372670, 86374923, 29020777, 74859444, 32750878, 26197091, 62345750, 70055219, 34628230, 4740779, 73673335, 77949025, 26314494, 42745946, 20282562, 38779792, 80442356, 72655997, 87004527, 73065934, 98967385, 59238708, 15873695, 4662881, 74497103, 54887325, 57839586, 99193414, 16455064, 6349822, 20303053, 15492164, 62793100, 55454916, 72178147, 19773032, 86818954, 94038113, 97648391, 61439291, 72187388, 92364420, 99602644, 59810527, 9615694, 98237136, 61287528, 3638353, 37926307, 19135888, 83978333, 65172114, 25070518, 92832347, 16198576, 62060381, 61799891, 7040728, 49731066, 39350225, 83528770, 74133403, 32725553, 82956431, 57105024, 71439291, 74513056, 78936052, 65777300, 11504176, 52893262]
config['test_seed'] = 18867274

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/city_scapes_c/pspnet_resnet50/single_task_depth_estimation_run_0"
