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
                'class': schedulers.WarmupLambda,
                'args': {},
            },
        },
    },
}

from runners import SupervisedSingleTaskTrainer
config['runner'] = SupervisedSingleTaskTrainer

# dataset config
from configs.common.datasets.change_detection.oscd import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.change_detection.fc_siam import model_config
config['model'] = model_config
config['model']['args']['arch'] = "FC-EF"
config['model']['args']['in_channels'] = 6

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config as optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 88646444
config['train_seeds'] = [2890645, 44973035, 39062169, 15415284, 53783486, 79018078, 17028550, 26234898, 51340955, 25221020, 97328543, 99655111, 70094698, 38929997, 95020421, 59260395, 56172943, 14920441, 77741169, 79329110, 36357611, 57576068, 40359852, 2498582, 66264750, 46552952, 29239796, 95505694, 89172794, 53582771, 23716142, 53964425, 34794513, 27027531, 63362534, 66020823, 71526587, 49477887, 69558377, 788428, 53810777, 55121292, 18899028, 85192642, 11967559, 40266442, 10142500, 47815406, 44599409, 44422491, 3322748, 47743301, 70362738, 95132495, 4545035, 96421720, 24953064, 52640226, 43679169, 78020891, 13629625, 25570592, 50412630, 99045171, 40871245, 87996871, 64396875, 89901865, 45774442, 22649156, 73014767, 10697956, 92497161, 56383242, 60367643, 9119022, 4818139, 98483474, 13239871, 36722232, 51371534, 25787650, 49697988, 36168125, 53991137, 27414962, 25619605, 22326976, 82504362, 44544040, 81429776, 77093936, 90897336, 88544230, 75531791, 76321035, 66387459, 40079457, 30247028, 44936036]

# work dir
config['work_dir'] = "./logs/benchmarks/change_detection/oscd/FC-EF_run_1"
