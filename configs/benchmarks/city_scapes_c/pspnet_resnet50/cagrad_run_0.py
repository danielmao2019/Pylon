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
from configs.common.optimizers.cagrad import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 82950044
config['train_seeds'] = [8837494, 8711823, 20334618, 66320770, 47451980, 28058174, 37398269, 7544577, 91553137, 49927709, 59862012, 91009080, 89753950, 92182555, 66622805, 44189878, 51213601, 75365953, 85898560, 1079938, 2381358, 31422621, 36076140, 16837496, 21896265, 5920594, 92906554, 99707010, 1056850, 52560820, 26329869, 9581645, 32000490, 33669090, 57166342, 47371562, 55141700, 32659964, 71334234, 95753363, 32135842, 57338832, 79119863, 99664442, 33825531, 72142215, 50980231, 13453657, 33452549, 58368540, 95572439, 314155, 65041487, 96191240, 87230850, 49847592, 57504, 71045397, 72559255, 54670033, 21982446, 78896567, 74957005, 44870383, 77943905, 42656445, 60046927, 41267610, 24187846, 58861015, 99869429, 12549095, 56621885, 80128963, 40337300, 21623605, 15589310, 24007239, 93067692, 32880552, 67090574, 42920124, 75629835, 22876438, 41472970, 79989254, 69622612, 47964572, 44868239, 33115435, 82489980, 41611261, 68237432, 62832652, 57015144, 40604956, 13366580, 76744451, 74910042, 55497748]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_c/pspnet_resnet50/cagrad_run_0"