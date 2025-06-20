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
from configs.common.datasets.multi_task_learning.multi_mnist import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.multi_mnist.multi_mnist_lenet5 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.alignedmtl_ub import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 14760303
config['train_seeds'] = [31973090, 67368994, 75343461, 40688926, 34360986, 36800657, 95881340, 43725872, 75518069, 7809126, 17704787, 5436529, 29273720, 94698781, 27768928, 90817728, 33979784, 41618505, 23581525, 7553909, 25212452, 74578929, 94790186, 79488647, 98193824, 79671353, 45913670, 212613, 60606706, 93157418, 99377223, 79863087, 67418193, 48548847, 15981892, 85290700, 9207264, 82872746, 18417167, 60156789, 21253836, 11297494, 21820060, 16021275, 26512122, 58567681, 80763936, 23774359, 50513963, 65251877, 81624064, 14725516, 6518029, 13420087, 72576705, 60855651, 64843279, 46777480, 74876989, 74066084, 22366632, 44802457, 56034082, 6898308, 22091480, 13051533, 16489731, 90565260, 25873789, 40114551, 71613779, 46199087, 42190249, 1610149, 51519639, 95248959, 15241135, 43222554, 90658510, 20214720, 59551411, 81531819, 23241617, 52944980, 33783913, 15922938, 11609768, 38496757, 2923146, 2272006, 99179255, 42367988, 89388087, 97286831, 94786769, 67443773, 33851379, 26610442, 81684544, 70089486]
config['val_seeds'] = [94322494, 9471860, 70443480, 71783971, 7558732, 24670232, 53102740, 34613109, 92219650, 35807792, 62556705, 30082292, 31430562, 48092902, 81299490, 47759151, 87629399, 79732531, 20244705, 79723776, 90422, 53356604, 97919979, 8501535, 45221240, 82177110, 23508687, 94419238, 52523803, 27438156, 17253308, 55307006, 64221616, 65840338, 15293685, 14507471, 60330981, 20247332, 66911356, 88666625, 84006115, 42241836, 3785197, 16425873, 94055464, 48290146, 81360951, 19918064, 57306649, 31988397, 18061707, 7803130, 54507358, 80366044, 66126025, 21884429, 18176224, 6596210, 79867428, 9767916, 465493, 4971047, 26721912, 51632712, 81721926, 86398533, 56860876, 83698258, 53548188, 66368385, 37285382, 45871217, 4489126, 45966633, 28081617, 1554457, 24275144, 2612428, 23702213, 44938996, 7420308, 35441076, 93367030, 10825993, 3106904, 4996562, 73991205, 16882451, 97047203, 33366740, 37798400, 80722013, 42646451, 60588750, 10716461, 23009590, 56085218, 24299884, 18765775, 6566278]
config['test_seed'] = 77853344

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/multi_mnist/multi_mnist_lenet5/alignedmtl_ub_run_1"
