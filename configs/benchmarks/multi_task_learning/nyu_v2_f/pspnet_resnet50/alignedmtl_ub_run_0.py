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
from configs.common.optimizers.multi_task_learning.alignedmtl_ub import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 65592137
config['train_seeds'] = [34975683, 89269568, 24142112, 4341283, 93127828, 62442315, 854421, 38748917, 72423750, 16861426, 62056181, 76611678, 27318450, 21461914, 69962467, 4399442, 93621152, 65712447, 74972379, 94033491, 43686597, 17416046, 9594390, 67317547, 50333390, 50843249, 87843313, 4827832, 66388912, 5697587, 39077477, 67894608, 26237239, 83590150, 73719956, 16966718, 28590511, 25315788, 47704488, 8703081, 58401156, 62051002, 94716612, 47401043, 75186768, 80597997, 53620170, 30692156, 12048196, 2008953, 11984341, 50993041, 56627321, 23812976, 42905476, 1530610, 16473621, 62010402, 69643054, 19919348, 8665788, 31438608, 95587152, 91773586, 40747545, 25783796, 73180194, 38004766, 3370497, 82912472, 74405111, 91711838, 28028483, 59913944, 15809700, 73741026, 65883649, 90705301, 62021356, 81715977, 58970570, 42529185, 3434644, 86276458, 99866439, 70069452, 55206403, 13450208, 47978753, 66654658, 74907167, 57928270, 53836413, 78955387, 87113034, 10605564, 58683182, 55878336, 24756019, 97202115]
config['val_seeds'] = [97115254, 76246032, 21211220, 33458617, 20942798, 63167108, 83587199, 604772, 7245307, 99418848, 43285459, 38805721, 93291840, 74843890, 71608718, 45886298, 98522316, 13934345, 45215857, 77807207, 31889397, 20605264, 43183287, 91102791, 42266409, 48269649, 49093168, 28706024, 30069140, 26939189, 47232790, 79927917, 72769829, 20848814, 5227395, 93105306, 52118300, 95927162, 94209949, 14197964, 68148700, 49649250, 17842268, 16777162, 41012195, 11668487, 10457182, 13775405, 69485797, 46092239, 78505593, 95661062, 11467656, 26020437, 65799422, 25943000, 50004633, 62438035, 59668743, 13134009, 9887472, 71841602, 81354890, 53115115, 5841064, 28805176, 23133199, 32034955, 39519132, 69270382, 21234463, 62166192, 70032795, 64789757, 73605231, 76592767, 78014552, 7951195, 76201676, 91938425, 68045000, 18591447, 53789322, 58478872, 40873729, 41699845, 52195536, 17071655, 63545056, 26879570, 61570253, 89636131, 53749833, 89137983, 14486737, 87960281, 62985969, 21838434, 44548353, 48368179]
config['test_seed'] = 29017126

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/nyu_v2_f/pspnet_resnet50/alignedmtl_ub_run_0"
