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
from configs.common.datasets.multi_task_learning.nyu_v2_f import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.nyu_v2_f.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.alignedmtl import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 83066893
config['train_seeds'] = [73279224, 72155881, 29349099, 26729157, 84634158, 30285644, 53578646, 57159019, 92479572, 59738153, 99910493, 30891351, 59195345, 13373932, 95090820, 199407, 80628892, 20023052, 39931901, 54976894, 78614812, 36321804, 44212913, 83721726, 12057153, 94644676, 77400856, 12571001, 20435259, 14161735, 20158304, 58153605, 94516145, 4068395, 48504557, 79220883, 96945991, 31832287, 22904380, 39326107, 78324236, 36305861, 41107732, 55327075, 87319350, 6205918, 58358385, 12615256, 31130549, 55056012, 83603707, 61197936, 94837752, 76841174, 16277968, 51585357, 50036184, 10672906, 52900672, 62589909, 72717629, 91445954, 18145971, 22500419, 93675325, 27756209, 37207854, 84751062, 14009977, 23785128, 32598974, 25253869, 17491015, 14746342, 59682263, 7249709, 82245942, 94504000, 21541233, 32238030, 51307796, 25581880, 3374776, 92909847, 8803133, 97607181, 16665064, 23946041, 5855393, 17272670, 53212064, 66821383, 9019644, 87753633, 17899053, 62396445, 48500200, 37216930, 17462718, 43113145]
config['val_seeds'] = [18334547, 3531366, 28327149, 42446302, 239584, 46914736, 67822262, 67840243, 69580936, 60653013, 1417162, 32582530, 75894353, 35064464, 10663682, 18030944, 23817342, 73700547, 90722343, 69881408, 61665218, 81170680, 38573450, 71088288, 88024406, 60578511, 32403085, 97255287, 7955542, 33782100, 39081272, 54203195, 32636121, 35398162, 46802375, 84837437, 58658530, 79230578, 90210119, 49838051, 65053480, 80133102, 51879006, 60181768, 77757569, 78927146, 47391704, 89185403, 96907068, 36902946, 98834258, 51159135, 29080089, 78943272, 15226029, 66343118, 54573475, 48974519, 14425099, 29706066, 53116688, 38661418, 79057320, 21225335, 34316697, 83662610, 33270627, 49100603, 7053742, 83004743, 50563579, 28201319, 52099464, 48418490, 3464467, 28624985, 9267079, 86983299, 52074331, 42864032, 67434930, 426638, 2007184, 99631777, 90246105, 83135640, 48047240, 7484351, 1910808, 17118358, 98518238, 78975513, 60950081, 18909328, 33848920, 59183824, 33501335, 937553, 93253590, 33553364]
config['test_seed'] = 18548506

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/nyu_v2_f/pspnet_resnet50/alignedmtl_run_0"
