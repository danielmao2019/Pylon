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
from configs.common.datasets.multi_task_learning.city_scapes_f import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.city_scapes_f.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.baseline import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 15161864
config['train_seeds'] = [67021429, 16325880, 15730930, 98774347, 96873156, 71957796, 18671568, 1008321, 29388276, 62520630, 2915593, 75334301, 98063511, 366745, 69280671, 98502204, 16746556, 7063146, 50155327, 80714806, 84943227, 72847913, 54032663, 14859515, 51812866, 88690298, 13109141, 65074901, 63955794, 30484463, 79371448, 8014838, 5492126, 10247851, 38899868, 27051672, 54717181, 67832750, 66875775, 87067253, 87060110, 12672222, 24449113, 27337003, 98055134, 43122252, 9250032, 49024654, 53824115, 82769623, 51111525, 13755016, 69345619, 19524196, 26109454, 30533060, 70556545, 6677499, 30270031, 25897079, 61060285, 37108862, 86168617, 76420597, 46172169, 67255756, 54454404, 99235700, 21227363, 65990663, 67386931, 41290262, 11369965, 46346264, 75119833, 97791975, 90043779, 40875103, 76641634, 83186228, 36137293, 3716338, 8706266, 94296461, 53443950, 50930395, 77372696, 28411140, 18856485, 65835070, 54796704, 83776688, 34383439, 5349918, 47663026, 59402871, 50819071, 10166169, 42568491, 38597814]
config['val_seeds'] = [77286255, 77986085, 61928297, 62667465, 24505530, 15702103, 13953294, 72244430, 24344374, 79114525, 95765183, 54781559, 29613339, 23964278, 23276380, 51566545, 93472768, 82377914, 34398771, 40130941, 71044061, 80184543, 81535654, 18921668, 62422535, 63608638, 67252936, 97800870, 62324170, 90287177, 5090426, 11236400, 70696503, 58900631, 76377, 72288559, 31135601, 40517051, 5910160, 14092807, 25022299, 19126762, 73620720, 7549182, 30920342, 71455459, 56031372, 26549021, 90930902, 30543300, 932630, 19299466, 78762413, 37361998, 58570511, 95481038, 587913, 57169405, 11720459, 49183226, 58589878, 95555513, 16505713, 71868466, 64432657, 49726286, 64968965, 70012054, 281684, 53166221, 7929587, 6751674, 75700552, 60583904, 73707647, 6102794, 3156005, 25271400, 56449321, 31470416, 71445360, 40695932, 3864443, 93453334, 53031756, 9760892, 29866792, 13363516, 12598619, 61970136, 56441666, 48014314, 42449363, 96335114, 27876170, 11583726, 1243586, 49176990, 85405652, 61799080]
config['test_seed'] = 24100937

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/city_scapes_f/pspnet_resnet50/baseline_run_2"
