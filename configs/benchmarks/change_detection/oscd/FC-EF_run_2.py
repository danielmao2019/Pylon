# This file is automatically generated by `./configs/benchmarks/change_detection/gen_oscd.py`.
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
                'class': schedulers.lr_lambdas.WarmupLambda,
                'args': {},
            },
        },
    },
}

from runners import SupervisedSingleTaskTrainer
config['runner'] = SupervisedSingleTaskTrainer

# dataset config
from configs.common.datasets.change_detection.train.oscd import config as train_dataset_config
config.update(train_dataset_config)
from configs.common.datasets.change_detection.val.oscd import config as val_dataset_config
config.update(val_dataset_config)

# model config
from configs.common.models.change_detection.fc_siam import model_config
config['model'] = model_config
config['model']['args']['arch'] = "FC-EF"
config['model']['args']['in_channels'] = 6

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config as optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 26933399
config['train_seeds'] = [53031702, 98499737, 22987814, 83083422, 43553997, 51739784, 3032926, 74450077, 26507404, 98027725, 21843346, 90757896, 36979679, 33320458, 53888240, 32899800, 42908090, 73506728, 66520780, 10016219, 50075728, 38113460, 57850607, 28928815, 85339471, 98677440, 11395149, 19820629, 41789405, 70922101, 72973246, 16338455, 24143908, 10743515, 90753908, 42396403, 52536858, 34732966, 10828628, 14802540, 20478963, 8440584, 31623247, 32275388, 16659214, 14178150, 54669771, 73430701, 60134909, 65674125, 33778036, 89824353, 84578701, 45179651, 99758489, 46112348, 44288144, 93480980, 18757003, 84505603, 52646620, 89557741, 25295479, 83658832, 67834729, 26786204, 41077364, 69007773, 95606686, 80513170, 85156543, 73086040, 66035512, 19583853, 98859758, 68944548, 35437106, 44756600, 21936906, 11131710, 38639317, 32261925, 23280331, 56635938, 29511073, 51892666, 40183297, 85994955, 33547213, 89285354, 15700945, 72997838, 40809028, 76110107, 58188305, 8873924, 82494571, 4229814, 49265076, 62858322]

# work dir
config['work_dir'] = "./logs/benchmarks/change_detection/oscd/FC-EF_run_2"
