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
from configs.common.datasets.multi_task_learning.celeb_a import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.celeb_a.celeb_a_resnet18 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.baseline import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 40881919
config['train_seeds'] = [26459594, 10105909, 49582302, 73466942, 4358157, 7397729, 3915434, 22932765, 45987191, 18474309, 7955807, 62745810, 77601820, 68352975, 95173454, 95143938, 71000584, 45493362, 25813413, 25577034, 40591791, 99512872, 67488645, 55188668, 66500127, 55898230, 98872379, 53459258, 61805328, 59822439, 80359995, 28524356, 12651512, 44906120, 71202606, 64869110, 6484331, 24513612, 98651594, 91068361, 90395623, 21318867, 56046433, 75153709, 52009393, 64810333, 84761249, 58303479, 5003704, 73387180, 75512064, 28481741, 6198947, 48659527, 49233039, 95532088, 36324602, 3503593, 9439530, 79895795, 65203907, 91517713, 8084745, 75203914, 91522532, 33913850, 70692302, 48350913, 31789645, 98394046, 17188358, 57751779, 94892200, 77391848, 69525167, 98694414, 98706388, 34655010, 97847724, 2786007, 11610512, 90076044, 29739426, 92100298, 81908695, 70016616, 35493386, 71490347, 88064035, 35327831, 26415425, 84847236, 34643021, 42612522, 23708771, 35723511, 98877802, 66723627, 96216857, 58509329]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/celeb_a/celeb_a_resnet18/baseline_run_1"
