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
from configs.common.optimizers.multi_task_learning.alignedmtl import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 8207692
config['train_seeds'] = [17235728, 97164175, 5980012, 2652262, 72127012, 88418750, 91615711, 65068151, 27817329, 58693173, 3989388, 75753589, 63211303, 75947939, 69076566, 60183024, 81311698, 21552231, 1098975, 46732164, 77965742, 62715263, 93538852, 68887007, 96826625, 47978538, 50019198, 17896990, 98524082, 33836970, 33720896, 36972373, 87072274, 46001011, 47636327, 32918580, 72600778, 77160006, 75727169, 27368940, 74751300, 583842, 9783012, 38577600, 11280947, 36253000, 10896911, 39778784, 91225509, 1589865, 85583092, 32856980, 52198767, 54771005, 87080589, 33506197, 12006708, 49233337, 80064979, 70626238, 87431276, 1541167, 34313629, 2852887, 33195770, 99470928, 7301362, 31067509, 59952984, 46787311, 20004815, 99096269, 38581746, 16431841, 8873164, 39077871, 66914152, 31543178, 58491032, 69806456, 8497911, 44711030, 80873143, 72086288, 60084786, 28512826, 79347893, 91032909, 63648546, 11999951, 95597174, 42197366, 17222104, 42671785, 54630200, 83467800, 84727186, 91674446, 46618811, 73467373]
config['val_seeds'] = [99103736, 29726664, 62210246, 806048, 54778887, 87989972, 1101680, 75787718, 40701563, 46840270, 67837930, 70868506, 23878048, 51474775, 29990564, 20057651, 66436155, 32479230, 92247408, 32254108, 54140003, 8245493, 98572339, 78865097, 27691678, 20920229, 94061327, 1921737, 49884476, 96802400, 30960394, 58035186, 80349272, 26488172, 65504795, 48253183, 75978422, 24477759, 88155757, 77854181, 99445710, 86603454, 83224231, 42171330, 42686040, 26997237, 41982290, 16342188, 83024008, 13730934, 32318564, 40374602, 63854884, 90678019, 69543985, 32215386, 52110355, 14802916, 9090606, 42026551, 48723457, 7023540, 37836454, 30065253, 72660267, 36198570, 77342846, 47129321, 95541171, 45728369, 94884726, 98171438, 77409385, 25133357, 24898087, 79750881, 77272836, 58009427, 3040237, 42999577, 59374536, 58286298, 79825438, 57363352, 97268263, 10997445, 57861036, 60479006, 22676938, 52821788, 37328459, 40723270, 69355990, 69695219, 64114575, 66335984, 89182480, 58361136, 66097846, 52745448]
config['test_seed'] = 25709842

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/city_scapes_f/pspnet_resnet50/alignedmtl_run_0"
