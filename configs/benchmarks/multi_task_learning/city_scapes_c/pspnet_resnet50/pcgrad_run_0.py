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
from configs.common.datasets.multi_task_learning.city_scapes_c import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.city_scapes_c.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.pcgrad import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 27519110
config['train_seeds'] = [69866977, 62771257, 2568140, 6990150, 47341067, 53312649, 92979453, 37136576, 48510638, 86722253, 56114624, 53023811, 4427099, 26220517, 71986047, 19461340, 12851232, 63263263, 65506178, 58952041, 12909657, 24438881, 61617644, 96040307, 5655520, 824381, 78707416, 38073428, 4503142, 27735822, 93812017, 2842483, 64898048, 80878235, 44677983, 91735567, 32558560, 17147212, 45496665, 42678143, 91103173, 21785061, 94972989, 19358810, 11953539, 77148219, 29311938, 76744823, 73837234, 38771400, 55677740, 88735824, 55845564, 96320749, 98326751, 59961169, 14062879, 94549364, 14931770, 62931163, 24351082, 13118299, 83769652, 41078544, 28212073, 5615260, 89016543, 9349119, 35957506, 62359420, 43899315, 32365448, 98530809, 44671220, 6889948, 40800986, 7457262, 23905045, 75636032, 15727208, 61863097, 76622544, 66896334, 61183369, 19592227, 32267139, 69833694, 26082946, 45535205, 37597657, 97394638, 18649283, 17276807, 44853002, 3289857, 76759095, 37330063, 88381301, 16648147, 21256529]
config['val_seeds'] = [36769510, 80727762, 64419471, 89986300, 88322025, 62817082, 69760829, 87051981, 97774063, 25589214, 20111420, 2279430, 77852944, 42758194, 70338806, 39172096, 62277160, 16752187, 76270288, 46317677, 23245383, 54258376, 19643002, 92912309, 2840973, 5120621, 61750141, 52331541, 36815296, 3018649, 91124456, 41235507, 88670542, 44474035, 6507760, 99646162, 77460606, 13185023, 25142055, 29917993, 58827098, 86146135, 24756793, 19762619, 76846542, 70752688, 79560531, 67925260, 13703146, 54958757, 6704218, 85208594, 75575656, 41372648, 40457127, 19514070, 67630813, 97247467, 9491615, 79944474, 13852097, 14708413, 852346, 51928814, 8531043, 24677268, 46948187, 3172147, 38208418, 56526373, 47238740, 86145654, 80951257, 29998778, 53012571, 78058889, 68046169, 48603198, 78091646, 27406459, 89733204, 83441887, 14487491, 94595650, 75492632, 62557962, 97678203, 33373046, 14804829, 41273021, 53995210, 25831713, 44682288, 31898469, 4134210, 28994052, 20966525, 18378053, 94012615, 1973055]
config['test_seed'] = 37637251

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/city_scapes_c/pspnet_resnet50/pcgrad_run_0"
