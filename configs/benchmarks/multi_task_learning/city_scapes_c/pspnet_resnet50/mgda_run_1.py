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
from configs.common.datasets.multi_task_learning.city_scapes_c import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.city_scapes_c.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.mgda import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 50515287
config['train_seeds'] = [96816866, 5827823, 90318612, 49674690, 99371628, 23577146, 23559246, 18111787, 85250850, 64718386, 4000444, 21038021, 28867116, 59677198, 69770888, 2375609, 68332304, 84284542, 82895182, 79581518, 85154084, 40134992, 54669597, 29228661, 70684196, 99878693, 96009091, 70111289, 17693159, 99328517, 6141557, 35381179, 96113917, 60678321, 62936071, 74293063, 29953, 12930769, 81399547, 94599363, 8697485, 11587679, 45039809, 32459393, 99707451, 89617200, 68329470, 83288454, 2534421, 72783429, 19908566, 95895815, 2177353, 72728807, 92083496, 34277919, 74623876, 37005468, 14337412, 72038573, 75213087, 85311120, 92730412, 11868049, 76926371, 76785307, 56828993, 94266128, 43845800, 30382438, 37493878, 98395266, 24583952, 53708529, 43960497, 43198181, 50358727, 85278357, 19997523, 76807887, 76346854, 84885518, 48523910, 78131821, 11280548, 71514248, 99050849, 10502398, 78249251, 99015078, 32792109, 84474019, 46382186, 49333771, 33278828, 3591144, 64837278, 98314866, 35257954, 27605747]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/city_scapes_c/pspnet_resnet50/mgda_run_1"
