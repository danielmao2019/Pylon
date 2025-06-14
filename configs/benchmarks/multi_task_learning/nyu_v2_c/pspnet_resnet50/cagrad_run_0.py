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
from configs.common.optimizers.multi_task_learning.cagrad import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 89108419
config['train_seeds'] = [72407205, 86877573, 68215388, 49323961, 28819869, 40035923, 96701211, 64756985, 68962649, 552767, 10652752, 73033885, 45006490, 15758282, 58804636, 50808608, 78720051, 33638, 73195875, 29824890, 82934510, 91617305, 7944760, 53171493, 95276993, 27191110, 29714076, 80029670, 1712258, 70280023, 63742363, 73458499, 70705994, 93161863, 89823602, 97674418, 51916446, 47083386, 89399403, 64234598, 48326971, 45743751, 49996587, 62105635, 55860135, 11898866, 76007863, 63504819, 52656862, 73632625, 2015135, 54304293, 1543286, 82942298, 75691300, 49095256, 42082249, 91504460, 21171661, 49086639, 19411004, 10864051, 11842697, 92211913, 72111112, 25859884, 84656631, 87502124, 94056188, 21649042, 59512583, 85545106, 41180431, 14455936, 935682, 82158700, 83830217, 31468724, 23517172, 54118412, 67396293, 2172849, 93676927, 94354067, 57826498, 83661959, 3927328, 77675139, 3587162, 93857978, 34321556, 198456, 90292336, 12651286, 23862793, 85500936, 73175316, 42917948, 83252720, 90076223]
config['val_seeds'] = [71077087, 87342306, 88801615, 58433828, 70850157, 54170438, 69919385, 36808577, 96068768, 44344558, 1640746, 79192441, 48795889, 96217363, 68765886, 85637926, 20302623, 2834913, 23244520, 76252282, 62008724, 72909897, 260834, 11660647, 10178179, 38671950, 96793708, 71423191, 25318367, 55521798, 32732542, 40498436, 27594841, 57391822, 95596477, 69344939, 25720318, 65815864, 27098160, 14190149, 79956139, 20929692, 59354527, 83260332, 73996049, 40215349, 18380839, 20475319, 26789164, 64155256, 14522256, 54358440, 37748702, 56102081, 91734705, 71030315, 93174581, 81769979, 22315618, 17596768, 44926279, 6232353, 9685912, 54285801, 35542133, 15892891, 70384783, 19933646, 68517494, 50644753, 38413438, 16533243, 7588509, 73576725, 64689778, 48742389, 32815920, 80894991, 7383936, 60671756, 69567344, 11303656, 73929275, 84362478, 2791049, 85116777, 40419803, 52554206, 88299363, 44670226, 12181821, 38727247, 8505888, 47506215, 3007995, 58965236, 96107915, 62060925, 11885908, 11873651]
config['test_seed'] = 61028569

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/nyu_v2_c/pspnet_resnet50/cagrad_run_0"
