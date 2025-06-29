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
from configs.common.optimizers.multi_task_learning.mgda import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 21978351
config['train_seeds'] = [18705612, 7103553, 99860674, 77789991, 18505518, 16308919, 33894329, 23235833, 19268237, 31748579, 26725270, 91091428, 22443516, 26921616, 39908557, 43860831, 99823063, 33423353, 84005767, 8678374, 43644817, 90244848, 19901265, 67191033, 28958927, 33617627, 58606653, 45978473, 93486807, 29483549, 89991795, 12694402, 35151819, 13062923, 33169202, 91929460, 37501787, 27562263, 21434150, 59025996, 11482020, 1100887, 93607018, 6304418, 93986202, 81530565, 93365834, 51408399, 84566212, 86145254, 87081466, 61824307, 11524413, 30282911, 73420750, 11135634, 11185047, 76319239, 10804620, 5732907, 99318657, 65271962, 10809665, 438644, 73671466, 87448660, 12850049, 63432903, 1097214, 32690064, 1796954, 22073457, 71057671, 45962765, 47592825, 57856389, 80298148, 90527109, 76238658, 20943148, 30374379, 52994604, 5021075, 33978524, 19541395, 82844124, 75471889, 48579242, 88189313, 90461672, 78035986, 9241656, 31382935, 9740458, 30731607, 64253186, 25860426, 26268635, 68818425, 93851653]
config['val_seeds'] = [30499682, 21531793, 61649252, 67810325, 57114188, 927206, 26860986, 20162011, 83834608, 63388294, 87960586, 36530, 4783283, 66568063, 7728937, 21452018, 57789433, 42966146, 21026981, 64401092, 12409395, 11217067, 96226922, 86828861, 69533343, 92471510, 23330774, 28243640, 60723737, 25895759, 81099611, 6604660, 90302817, 81063324, 11484813, 3625433, 54329758, 62273350, 77655849, 46138982, 91050022, 98212935, 1339144, 78884653, 57680290, 81624119, 27780613, 16328917, 76655747, 19635185, 31313953, 28752575, 60098077, 21364448, 47250132, 88521460, 86262029, 4963058, 23948649, 50832342, 24959323, 7519653, 24181280, 11386293, 486244, 15043001, 37433369, 7285874, 6318021, 26968892, 63467822, 15491183, 32279394, 38380373, 34671598, 20930778, 3395726, 49305108, 46281704, 68885913, 85333434, 12931042, 23819717, 11147947, 22314731, 93252530, 95154170, 84192906, 85258074, 77665959, 24913569, 95320865, 96059017, 97216229, 67625070, 37745526, 46578920, 11983911, 71802159, 65493957]
config['test_seed'] = 68032233

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/multi_mnist/multi_mnist_lenet5/mgda_run_2"
