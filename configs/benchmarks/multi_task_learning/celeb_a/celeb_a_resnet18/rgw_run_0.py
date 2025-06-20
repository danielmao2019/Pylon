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
from configs.common.datasets.multi_task_learning.celeb_a import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.celeb_a.celeb_a_resnet18 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.rgw import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 6982607
config['train_seeds'] = [16192096, 4247193, 6238228, 99281965, 22561176, 67334008, 86968974, 56035704, 53599187, 77105828, 30378932, 88617301, 70896805, 99189317, 77196575, 83952615, 81916848, 72483925, 31094913, 98353992, 3118812, 39864075, 60907994, 10987140, 85294784, 50677685, 88415471, 19684632, 42954311, 39768404, 973577, 3354786, 12733223, 13243027, 57266415, 95428659, 6497074, 87038009, 15260025, 56436744, 54556305, 14009949, 11520265, 23187242, 71089750, 60458333, 50665674, 30447556, 3730072, 17732453, 10771155, 5156715, 74154053, 6111478, 70076779, 43891289, 31538176, 97970534, 38179334, 22195576, 86729439, 13608866, 10325944, 58534236, 43010592, 40911026, 52931184, 52587681, 46080169, 48242936, 78662199, 68645758, 22154059, 18626752, 29170303, 80905047, 20880761, 46456606, 91161489, 76479064, 71198461, 83062237, 38597031, 98608219, 33270713, 14673060, 40469623, 8136827, 79156758, 34993490, 268506, 23479525, 68072067, 24511216, 68476600, 57510275, 20628826, 29528721, 23617418, 23845261]
config['val_seeds'] = [63782944, 28290069, 82000119, 62145077, 63238641, 75850615, 80686148, 38220266, 7202141, 61066984, 29206127, 17232224, 14761353, 72269992, 31901143, 98678730, 31787798, 84901836, 55305275, 24989992, 87428500, 3754224, 96560113, 65533801, 48893132, 61617111, 42056416, 25858511, 423734, 17948274, 2189436, 88871323, 29609059, 94133726, 49968108, 28383235, 22493370, 73168848, 14708298, 89329312, 73435139, 64928190, 72192932, 60276430, 65443141, 5659732, 74915322, 80278076, 39872828, 98106554, 97408626, 58977763, 51849452, 9807558, 88294562, 56144701, 73332485, 82705033, 11379432, 27935483, 22897672, 74650999, 42819670, 48128121, 6504758, 27328883, 79591251, 54874415, 47736322, 70845773, 5913378, 56888542, 84294827, 51992036, 99420822, 18363166, 82280671, 33531845, 12290860, 66367184, 75992916, 67627693, 96116913, 90943566, 77906206, 2262343, 71174336, 83629558, 46974922, 56464219, 97830472, 42245145, 89123202, 20777292, 38418836, 91552797, 68556806, 31531655, 44352272, 63864280]
config['test_seed'] = 75197535

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/celeb_a/celeb_a_resnet18/rgw_run_0"
