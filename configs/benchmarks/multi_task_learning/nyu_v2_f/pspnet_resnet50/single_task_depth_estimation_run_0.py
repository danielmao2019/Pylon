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
from configs.common.datasets.multi_task_learning.nyu_v2_f import config as dataset_config
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
from configs.common.models.multi_task_learning.nyu_v2_f.pspnet_resnet50 import model_config_depth_estimation as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config as optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 97181974
config['train_seeds'] = [4468822, 2084434, 33548769, 58109320, 78623474, 14538593, 66651361, 38352536, 64413111, 74369457, 89759943, 49427474, 51455776, 60205272, 89232244, 74223884, 63030301, 7872224, 70142619, 63550057, 22646335, 69804902, 54579071, 64051858, 26742919, 41140042, 2029032, 77267810, 28480942, 33454930, 56077417, 49153011, 52282415, 85661501, 52131001, 51528273, 2660682, 40230348, 39349627, 51984249, 82129838, 75956172, 87858540, 31014333, 14534325, 28611096, 93959022, 82092998, 61829536, 39757508, 24475957, 22184352, 8661841, 5976887, 47242685, 45200223, 99046062, 59523698, 587986, 99251216, 47414995, 57232043, 85848752, 68303917, 55133982, 48369315, 70393335, 99643022, 5904366, 72984824, 80412145, 43617192, 41089077, 99231423, 14834319, 94490865, 25405108, 8640061, 68373424, 70786450, 966766, 21419709, 80119088, 85060453, 77724649, 63736399, 65197631, 83779550, 6779337, 92833132, 20119510, 70417581, 91664578, 95973675, 51414226, 10646981, 18217497, 18212207, 29347444, 31760370]
config['val_seeds'] = [76149546, 75479725, 45743623, 29611231, 26141236, 54730086, 38482608, 33878799, 51916740, 97567821, 87745167, 70055500, 2317210, 56163819, 93771221, 15141557, 66937688, 74620080, 89718873, 87365929, 39156760, 45240307, 57966197, 89104694, 2906413, 13762111, 41820089, 86879041, 76389467, 31443932, 11126677, 53041428, 2015566, 4417514, 58283119, 7552784, 25346446, 29500582, 84471270, 62545741, 57643260, 80664217, 32766567, 23616585, 1728557, 46905807, 29627120, 54854504, 28032004, 15358364, 57251151, 56229646, 5549884, 79397489, 26230393, 63489844, 55148299, 64010689, 12267096, 28972643, 79847650, 4905910, 50446822, 76572958, 95201807, 94877180, 80708789, 13605647, 87365312, 45116886, 39095685, 87849045, 98963957, 12561819, 75851867, 3076582, 1716766, 98436088, 7419092, 30099935, 94975044, 22534656, 51858496, 10857161, 50467181, 78845741, 56055511, 28948618, 49910423, 63993324, 37837711, 62980912, 81473641, 60480666, 41611474, 52361144, 46844191, 98837265, 93282455, 65459666]
config['test_seed'] = 78389715

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/nyu_v2_f/pspnet_resnet50/single_task_depth_estimation_run_0"
