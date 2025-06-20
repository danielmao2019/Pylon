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
from configs.common.optimizers.multi_task_learning.rgw import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 17574091
config['train_seeds'] = [35932732, 42915980, 26635195, 3561558, 92762671, 50879388, 10769529, 72179000, 34692146, 97657268, 92336819, 54398752, 20801218, 20731249, 80050565, 50623028, 93110248, 16999986, 73507303, 98207931, 69222180, 75899910, 1962989, 21715802, 15054914, 43561351, 54222131, 52848231, 27968500, 43497576, 81965033, 50778518, 72473873, 48606629, 30678204, 89345046, 54678069, 44062855, 45272630, 21104485, 40242153, 47702398, 37876871, 17168847, 167668, 47237177, 86335979, 42027553, 38621692, 81344111, 89697473, 90447952, 70849550, 70205888, 3875052, 57619977, 74789375, 54329463, 98430232, 74652297, 65388429, 45633161, 14903598, 16207002, 20780631, 57168014, 56233106, 61834268, 30721481, 65344806, 70468396, 76460369, 15679906, 57381754, 75656461, 77371695, 83505528, 26630419, 69445639, 22616442, 48229794, 9617398, 7854479, 50238937, 69484291, 7000577, 46932926, 44704617, 56725550, 59921180, 44129286, 16695546, 20313611, 87082148, 66999319, 81081843, 78934498, 57911978, 15398955, 94521172]
config['val_seeds'] = [29957333, 56112117, 83541104, 97648139, 94133196, 76523107, 50949585, 45206858, 7105474, 99446626, 48951252, 58155920, 85885050, 40300543, 75230496, 62721939, 38811136, 68165775, 78056499, 96956482, 41114313, 66922833, 81692365, 25572556, 5850311, 50995421, 54640189, 35380877, 7502953, 78285244, 57469639, 60185330, 65895749, 67312745, 43084955, 15600843, 93619115, 42491154, 2961697, 72014076, 85968291, 84596194, 16171806, 28151011, 34706051, 23930059, 4273305, 34028404, 51166653, 82045259, 94578028, 10287146, 84145243, 54057672, 46882910, 56764702, 28690028, 88359899, 42176347, 60821617, 53260253, 75535859, 89616728, 94873635, 36858101, 28614633, 98342544, 84270876, 68247792, 42823931, 31773857, 58793536, 45340960, 25378380, 48074300, 31342518, 75872762, 18342250, 99370840, 45822746, 25163451, 25694207, 27452338, 98095154, 80190935, 53456968, 21955422, 86041611, 79893194, 41191962, 15315990, 27746078, 37580247, 71660684, 31959068, 7062916, 48644508, 74432496, 5697687, 98524205]
config['test_seed'] = 7414906

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/city_scapes_f/pspnet_resnet50/rgw_run_1"
