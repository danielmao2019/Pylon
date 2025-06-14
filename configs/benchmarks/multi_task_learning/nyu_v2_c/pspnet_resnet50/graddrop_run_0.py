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
    'val_seeds': None,
    'test_seed': None,
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

from runners import SupervisedMultiTaskTrainer
config['runner'] = SupervisedMultiTaskTrainer

# dataset config
from configs.common.datasets.multi_task_learning.nyu_v2_c import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.nyu_v2_c.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.graddrop import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 12173166
config['train_seeds'] = [79711098, 9939970, 97206999, 95542336, 32933834, 31867585, 27931642, 73248009, 87014505, 30096648, 45625249, 12740758, 91652034, 26383595, 19018864, 16685503, 51535598, 60549657, 487808, 36459266, 45988816, 49321598, 10572706, 65028064, 91923031, 11887071, 82445199, 71263964, 34511126, 29486589, 68228374, 42641840, 30669262, 91767360, 73314786, 62185636, 6754809, 4785867, 94689725, 20684955, 15111747, 7879213, 7476844, 6525226, 89791606, 24192527, 78721813, 29660698, 35855337, 99971727, 57987085, 48532830, 21192765, 31033628, 35527871, 52182951, 90396787, 18151062, 24592666, 83292861, 20861719, 683141, 72349290, 21929134, 30256577, 51153538, 13232671, 66602434, 84875089, 24644913, 48355403, 80678288, 73315701, 99734933, 23641478, 53466209, 35093090, 53899950, 1327654, 8218749, 27025024, 38716589, 89779251, 50933214, 51976768, 74188103, 44045640, 63141038, 43061696, 95456882, 70929668, 67750518, 59063141, 30839525, 50917616, 69000944, 97881506, 50462065, 53154854, 95049244]
config['val_seeds'] = [5196353, 45503017, 6226949, 35824881, 97819327, 30030510, 89058384, 29651125, 57235464, 9698451, 42398165, 84495943, 89744309, 24816942, 94918239, 36315495, 49364599, 2028537, 81029214, 81691912, 25794860, 34906550, 57339945, 6615894, 73914436, 67975605, 5777595, 91484766, 71889183, 61708940, 14977724, 98742553, 684325, 1661476, 95717021, 69774416, 66169487, 10808987, 52317442, 61032351, 88169099, 21286370, 58974257, 7478479, 45095489, 7545323, 68827432, 1535795, 25253918, 35823855, 18509199, 76219298, 99172134, 81776141, 49719689, 42986213, 43674315, 58815708, 54831610, 40801227, 44746806, 78973762, 34636161, 90163732, 71737556, 89949771, 12056889, 89040497, 98931691, 80621116, 10827647, 15083601, 63888260, 12254528, 48878826, 30364044, 63166846, 43567587, 5795484, 37594781, 5943762, 57511403, 43396164, 27782960, 93945288, 37210150, 37089931, 38654861, 34839228, 97261373, 82051234, 89206496, 60505582, 44599362, 6842468, 51003542, 76481776, 38359719, 92899270, 88219088]
config['test_seed'] = 66047411

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/nyu_v2_c/pspnet_resnet50/graddrop_run_0"
