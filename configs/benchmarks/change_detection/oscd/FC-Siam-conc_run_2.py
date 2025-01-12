# This file is automatically generated by `./configs/benchmarks/change_detection/gen.py`.
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

from runners import SupervisedSingleTaskTrainer
config['runner'] = SupervisedSingleTaskTrainer

# dataset config
import data
from configs.common.datasets.change_detection.oscd import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.change_detection.fc_siam import model_config
config['model'] = model_configconfig['model']['args']['arch'] = "FC-Siam-conc"

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config
config['optimizer'] = single_task_optimizer_config

# seeds
config['init_seed'] = 77090730
config['train_seeds'] = [78424897, 73668235, 36987133, 66386206, 3481123, 89489318, 65317694, 17691972, 5084693, 45739637, 79430106, 56576547, 2551527, 56652887, 68214687, 10629929, 77160650, 95206003, 26770131, 60096705, 15967253, 99537671, 98310581, 20484456, 45898331, 11323136, 66770204, 12579319, 86015026, 9977688, 91695348, 23555461, 4862318, 97975423, 87539630, 82687526, 30057959, 82978271, 7968540, 37362802, 54040837, 59849742, 35621053, 57844867, 96213442, 1374950, 93221220, 90287995, 88015650, 5262720, 24238, 83416868, 74891564, 63268388, 80047391, 60465095, 86861870, 38074937, 29151679, 30658415, 43218191, 87290723, 40944563, 29110391, 94630670, 28757003, 51302294, 44927842, 80087051, 55349629, 58626300, 24503625, 75445456, 24239477, 62934375, 17117811, 21929233, 17910261, 27057392, 24562047, 2807691, 51676682, 87125783, 30518629, 55363013, 45899157, 88299350, 22978946, 81197930, 28161347, 90611646, 89910914, 88261067, 96418545, 24972006, 44699023, 43108704, 18786604, 31987843, 35469921]

# work dir
config['work_dir'] = "./logs/benchmarks/change_detection/oscd/FC-Siam-conc_run_2"
