# This file is automatically generated by `./configs/benchmarks/multi_task_learning.py`.
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
                'class': schedulers.WarmupLambda,
                'args': {},
            },
        },
    },
}

from runners import SupervisedMultiTaskTrainer
config['runner'] = SupervisedMultiTaskTrainer

# dataset config
from configs.common.datasets.nyu_v2_f import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.nyu_v2_f.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.pcgrad import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 26078925
config['train_seeds'] = [73142738, 81417674, 83582649, 30556053, 51054476, 85879832, 10857226, 18417910, 31088849, 48555117, 77437010, 53441536, 12621960, 71747862, 94934198, 16311224, 84741949, 62680838, 82342375, 65323239, 23902851, 31685673, 60168679, 4246549, 3859380, 70461061, 97019304, 12213358, 68685795, 99810679, 7227952, 1327551, 7334311, 2996860, 89880881, 93938449, 25504709, 68816792, 804284, 38644209, 93834502, 74743598, 24710829, 23720572, 51612227, 96059945, 79569448, 34909063, 64984, 7857411, 3310121, 96611477, 86260216, 63037032, 15647717, 27566097, 47101069, 2092919, 866930, 72262903, 42980611, 18927736, 30553412, 60985455, 56913493, 80818007, 71909887, 30033990, 67756224, 95702291, 90194688, 63500604, 48022965, 84294149, 94177821, 32463042, 60865632, 38883371, 61125078, 65296276, 44682549, 44129308, 3240798, 22399994, 51163987, 55557535, 69655515, 87958139, 76975772, 57788242, 86576515, 1342535, 7220922, 75042310, 36784869, 55551482, 68102068, 6291046, 82743974, 90003373]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_f/pspnet_resnet50/pcgrad_run_2"