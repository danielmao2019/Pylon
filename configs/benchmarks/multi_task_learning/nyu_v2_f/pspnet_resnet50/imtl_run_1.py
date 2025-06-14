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

from runners import SupervisedMultiTaskTrainer
config['runner'] = SupervisedMultiTaskTrainer

# dataset config
from configs.common.datasets.multi_task_learning.nyu_v2_f import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.nyu_v2_f.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.imtl import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 12474961
config['train_seeds'] = [68762149, 25962464, 9576209, 27595367, 39008577, 95188254, 17319707, 46128990, 97026662, 85136810, 77194020, 30418374, 80691664, 9997243, 4406572, 57776916, 32942859, 81141727, 54608554, 23433976, 4403547, 85305907, 96277469, 88556576, 46108770, 22417178, 43762285, 55768160, 9901587, 90598997, 45907536, 67864073, 60392658, 57240946, 98942158, 75692286, 53773747, 17725464, 63627685, 88466444, 73195561, 13368583, 43840837, 78837847, 23703641, 67141865, 48558640, 87033916, 96835123, 25332659, 44968007, 85038464, 79256362, 45946684, 21331470, 40759095, 27853053, 34543752, 13609745, 55934527, 34624007, 71257202, 82644376, 67903445, 64567043, 10459304, 53281378, 51442760, 37264581, 84315738, 30407997, 91165560, 45090029, 72476257, 82585881, 20528262, 48693949, 27070411, 3184130, 57448865, 49834907, 54753490, 26530816, 51600058, 10331580, 18518244, 51202545, 36831292, 51383958, 27149537, 26909762, 58808420, 38489390, 20393454, 6223953, 32332523, 84776376, 3190636, 15242894, 85756037]
config['val_seeds'] = [34852810, 54970682, 77234883, 6359129, 6492448, 51975230, 49365458, 23380534, 70965704, 271617, 5591920, 47017990, 31282060, 54933359, 66870677, 64101352, 12014124, 75798922, 55092805, 49343928, 69078864, 90516099, 32245145, 90490295, 33784801, 63656903, 84384083, 62735019, 45600140, 8916754, 31881740, 77425489, 11814165, 12053044, 32254108, 63184837, 89954520, 27451614, 26376387, 80464139, 68405898, 40144951, 97005463, 77919776, 41277706, 5475080, 79185406, 42401209, 7197935, 69646978, 3861591, 30285089, 73066698, 53970483, 96811960, 67137670, 90269487, 41672261, 60714211, 83041309, 26293814, 66547345, 98905808, 93555050, 85066372, 31287167, 13155385, 19301907, 24768973, 15058061, 38187544, 35690645, 57364700, 86300749, 74269100, 28176011, 17471999, 77706903, 29517091, 60496069, 76881877, 22479142, 57190751, 16499373, 94756852, 81837671, 1976787, 78272209, 15639312, 39660091, 7140531, 32507592, 99705489, 34900140, 7603720, 36520932, 8532661, 75627083, 22561432, 3467881]
config['test_seed'] = 23691806

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/nyu_v2_f/pspnet_resnet50/imtl_run_1"
