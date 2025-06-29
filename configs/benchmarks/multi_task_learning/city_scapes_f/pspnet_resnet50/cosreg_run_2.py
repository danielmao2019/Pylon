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
from configs.common.optimizers.multi_task_learning.cosreg import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 8032445
config['train_seeds'] = [405822, 12914947, 37393662, 75082217, 90936822, 18470940, 11566499, 43043279, 41963107, 69144768, 85746762, 5488245, 63465581, 86866302, 97403154, 53649527, 78800259, 84463871, 30186247, 68008881, 25574249, 44807497, 34788947, 38942598, 25883948, 94222952, 44466718, 93894228, 40369330, 23520337, 65379055, 41047502, 36372827, 92879846, 82512649, 88968979, 82241075, 79676237, 12666255, 79696573, 23527117, 53148040, 18569785, 11217935, 32869354, 71966491, 18674981, 65638630, 83011435, 6918417, 51039808, 54327183, 80188913, 24531297, 42647795, 75090498, 79415831, 15741172, 29007563, 52898216, 5123954, 91245309, 76188784, 15782991, 62137119, 49389113, 69177257, 11276066, 3578372, 84368566, 88468155, 53701079, 57162082, 43449141, 36714577, 35684750, 49950375, 90432657, 50364865, 37970153, 22526078, 93191400, 70351777, 16901502, 69763510, 12104595, 97859743, 95499454, 12380550, 28837702, 15575923, 19649645, 35663464, 68323406, 69244317, 30878093, 53635028, 83359012, 54386211, 77789931]
config['val_seeds'] = [91519189, 11141151, 32059844, 76499749, 39485683, 12863333, 70379775, 9408352, 78964898, 36912724, 14099320, 30793045, 5488387, 52106215, 13278156, 81920981, 69660379, 56504566, 95109200, 24325939, 29775907, 81574472, 99951592, 53287769, 486895, 82135862, 22014797, 83723243, 69601146, 63947440, 37870821, 77368461, 50498575, 67308547, 27515951, 46275676, 94521353, 4595006, 63343885, 78042708, 22821115, 54856367, 66285581, 69990871, 23428673, 55217957, 70171379, 35992647, 70713188, 86813974, 83018630, 11344350, 91332908, 94498135, 52002084, 7115929, 36327494, 34398347, 49715260, 20051219, 3182627, 13945498, 95754317, 80746699, 51935076, 8603530, 45262504, 7305029, 71030279, 29597529, 18228959, 36104747, 93940601, 70420085, 20577505, 78493967, 55524017, 70376063, 71400115, 74143288, 55904094, 36031067, 67338535, 37036974, 86171062, 77513040, 92782239, 95677008, 62666429, 26220581, 74948732, 3440440, 25686215, 88218883, 33702716, 66126754, 34045204, 91991337, 80273770, 39617517]
config['test_seed'] = 81002271

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/city_scapes_f/pspnet_resnet50/cosreg_run_2"
