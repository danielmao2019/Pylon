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
from configs.common.datasets.city_scapes_f import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.city_scapes_f.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.cosreg import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 75045376
config['train_seeds'] = [3709534, 75731001, 93820798, 35483611, 51077895, 83231041, 90210644, 81009438, 89330780, 82124614, 55421431, 2645945, 15371290, 20141498, 17149293, 27819014, 88336381, 25556916, 64152160, 72955877, 62584121, 28244056, 30784768, 62932781, 57825462, 83482320, 49508283, 12304690, 90866219, 71375381, 5989086, 66459044, 78980524, 52878995, 60888649, 35110803, 35684623, 19797111, 54902006, 36268493, 96560021, 758704, 14282538, 9479771, 39652003, 76550445, 54218989, 59167743, 77601736, 85203811, 18099513, 46966886, 53062137, 34629549, 6594069, 73521915, 39600843, 19553627, 49669963, 82166650, 90090757, 5315402, 41828099, 55207246, 14421530, 87617767, 5599827, 90824945, 52053099, 71970283, 99162626, 7627463, 520903, 73832062, 12809649, 84917004, 99429051, 92353059, 81664087, 54011011, 98201944, 82756076, 56037514, 48758593, 12015414, 92138328, 53071896, 84305996, 12562482, 3042613, 64219457, 69683292, 53143743, 99942838, 86991402, 3762213, 17522550, 20496989, 39993884, 76147765]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_f/pspnet_resnet50/cosreg_run_1"
