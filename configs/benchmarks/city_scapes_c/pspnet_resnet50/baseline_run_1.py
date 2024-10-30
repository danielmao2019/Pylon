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
from configs.common.datasets.city_scapes_c import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.city_scapes_c.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.baseline import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 59485890
config['train_seeds'] = [65869740, 34106052, 10853911, 63720514, 36764661, 51106988, 11991771, 39259114, 91015877, 13072277, 91470411, 49882243, 93481597, 3915496, 32070631, 11630817, 21337258, 99003079, 33375996, 42501856, 5015701, 55188244, 46852865, 19035138, 18089110, 8729401, 60230737, 17143173, 49217296, 113341, 72841815, 62057489, 97130117, 63625862, 83570857, 11288915, 22258362, 91717329, 77177086, 35634923, 59316880, 96774226, 51479328, 91689534, 21780802, 7832426, 19903154, 28380079, 33083714, 10822971, 71428759, 23820188, 76841066, 47878343, 33930619, 12982724, 20951362, 65993699, 71823342, 54712370, 29938045, 24414651, 30876572, 25566895, 40144321, 3811721, 53786462, 3948939, 44534430, 60077735, 91457954, 72901353, 52854088, 96544555, 16498874, 61699989, 29672082, 68975866, 71558103, 65372923, 62628119, 43013699, 30545465, 17984740, 67431159, 26601620, 20456660, 16884036, 78137461, 50180076, 78666496, 89328528, 60755120, 41928992, 30063464, 31943873, 43265518, 64977437, 14075948, 77968386]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_c/pspnet_resnet50/baseline_run_1"