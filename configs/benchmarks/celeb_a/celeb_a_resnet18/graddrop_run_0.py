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
from configs.common.datasets.celeb_a import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.celeb_a.celeb_a_resnet18 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.graddrop import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 4010971
config['train_seeds'] = [3878547, 82432676, 40339489, 12345793, 27917039, 75513365, 96547553, 39819035, 44167326, 36582133, 38211843, 86603337, 45345986, 17918139, 65464411, 27371123, 16872813, 43068515, 12113542, 70473271, 38944243, 2933476, 80353456, 15944188, 84728231, 62122042, 56301260, 49602763, 86654165, 98943063, 88770461, 44975773, 62216305, 57331828, 91456260, 63220422, 5853065, 28926619, 87322476, 43158890, 47735009, 82356339, 13724186, 93331706, 17610856, 26333627, 48569493, 91094709, 15021711, 28825745, 1627863, 66916727, 63287751, 76649269, 82253841, 85567174, 53069871, 13859936, 86044504, 71477507, 20141397, 78982672, 24618487, 69907698, 66242545, 45061910, 95388383, 58490492, 21136195, 42022831, 20862359, 13574190, 6459618, 37420050, 96977462, 82326882, 22418818, 59322187, 11920273, 4952690, 18660182, 46488074, 87933360, 36116164, 99828500, 14819990, 91002189, 39231263, 31000368, 67132486, 5500601, 22369138, 84017165, 16608284, 42386698, 57147118, 35500470, 81617547, 8888953, 28739888]

# work dir
config['work_dir'] = "./logs/benchmarks/celeb_a/celeb_a_resnet18/graddrop_run_0"
