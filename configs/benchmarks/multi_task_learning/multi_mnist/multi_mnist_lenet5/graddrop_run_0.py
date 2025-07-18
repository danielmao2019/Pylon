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
from configs.common.datasets.multi_task_learning.multi_mnist import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.multi_mnist.multi_mnist_lenet5 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.graddrop import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 59769072
config['train_seeds'] = [30862157, 64027670, 60105508, 63032441, 47435904, 80365802, 20244501, 16779066, 25511962, 46574133, 1227794, 60879876, 48824128, 5183461, 85873670, 69912221, 67245978, 40397487, 9989411, 88129781, 33847490, 22973525, 7454693, 35540897, 86666473, 5309743, 50436074, 72917088, 531253, 39370427, 73225468, 42756968, 34013632, 88667516, 73165965, 43219734, 74334350, 47131112, 20973756, 80835703, 64500227, 39865859, 32665260, 89345193, 16849480, 84519774, 61549319, 98300750, 77679466, 38212207, 17448806, 9890974, 15533607, 81964273, 51606492, 33405277, 91863457, 21643516, 35527977, 1135264, 59938800, 70593669, 92425520, 47202410, 19061788, 9092531, 63874128, 78254064, 58980739, 29866790, 9014119, 4651231, 28894489, 15439078, 62165057, 68228396, 5565120, 61925865, 56866139, 91637903, 35923534, 70820954, 28203493, 40707711, 10910933, 48113347, 36879560, 81627967, 80771292, 42703092, 31192277, 6383342, 85627239, 94896907, 97566151, 61362422, 56729876, 98492665, 51054957, 66262826]
config['val_seeds'] = [96935854, 11787367, 4182258, 41865801, 38734692, 73695172, 95140244, 9933078, 37895028, 50292574, 54965548, 5179238, 9933700, 96665477, 58440650, 96712135, 82100174, 34375894, 35868317, 77931176, 72991304, 63339121, 8605952, 32091463, 10845007, 63358971, 12668283, 27974425, 29850345, 52689032, 86319826, 43994453, 11674305, 1782169, 75811446, 41601886, 58427946, 95021071, 50530842, 65211489, 42292121, 46386066, 47266945, 14959275, 18943525, 32938811, 76887058, 27317700, 80450741, 68275682, 27523240, 18378795, 27775496, 22330490, 36508189, 98642759, 9205144, 38937893, 18420335, 52523637, 37879322, 43214699, 33745766, 34698556, 623691, 89219194, 5727244, 66736752, 65283969, 85962855, 684249, 87376503, 29891719, 78221267, 6192848, 28589671, 61547074, 76599088, 81368248, 50805077, 89179456, 25568589, 73661404, 33330199, 28936750, 99877533, 87548761, 28391169, 41478987, 16047091, 13061421, 34469291, 24917819, 32104633, 86023477, 76485885, 93031441, 87909949, 17595043, 95519238]
config['test_seed'] = 4212466

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/multi_mnist/multi_mnist_lenet5/graddrop_run_0"
