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
from configs.common.optimizers.multi_task_learning.alignedmtl_ub import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 50132036
config['train_seeds'] = [13600588, 1778886, 22583934, 73077075, 84195599, 9193216, 59001277, 74101383, 29364982, 20791859, 10068904, 73223394, 45805472, 23592075, 50278436, 59120635, 86513604, 75125841, 55048536, 60785407, 33023250, 66456946, 87791068, 64826379, 99160764, 80036862, 87783033, 48501896, 7688505, 19789646, 39892454, 39220224, 29469798, 54907760, 18354092, 12025083, 10607617, 47175656, 17573181, 39965156, 49387878, 76277429, 46982486, 29985870, 11691692, 18640398, 15788433, 70603591, 82621397, 27843784, 56841352, 63428456, 9754364, 26400452, 84359528, 31763068, 91469392, 96306189, 15314149, 39665390, 87071173, 91542874, 26640125, 29110677, 87271466, 54192567, 4502644, 1721841, 84424451, 17676995, 22659720, 22184219, 86481717, 11532667, 13040038, 87437071, 5170306, 64051742, 68846467, 94758605, 59418164, 27943393, 39101176, 59962704, 1212421, 84536900, 12460688, 59927747, 63791593, 38872745, 95194626, 21976167, 84490733, 5479721, 28335085, 58807583, 42470207, 64100315, 42172543, 40537340]
config['val_seeds'] = [80625246, 98921763, 78186606, 32148726, 92775173, 58781881, 24387550, 24882462, 33041917, 95871907, 53714816, 17246201, 780923, 806767, 92755769, 5319699, 85152093, 32297040, 30899740, 78499081, 99945752, 62756469, 80541164, 42243611, 71746326, 40036966, 16156265, 58160264, 84108983, 84992320, 26920152, 33695526, 49320501, 79815403, 12318508, 10615801, 79353404, 95584755, 17911505, 42837249, 66598208, 57275697, 85292593, 50803420, 78650613, 80603011, 75685227, 86246034, 3423338, 90341812, 1408912, 66433371, 57589102, 10896812, 59356358, 58683818, 24309826, 97022244, 40633348, 5484128, 32254040, 14646952, 10387995, 47992791, 29250357, 15606469, 69553132, 35160483, 14080281, 74850029, 3423755, 9957040, 77498662, 22986011, 68652613, 99717892, 66112506, 7431755, 58675746, 21679031, 14047491, 73592280, 80187494, 2918347, 4598170, 57633220, 98237163, 92743600, 36637023, 60799010, 54948588, 41405525, 2627068, 22170061, 89983521, 72269650, 47183033, 95296746, 63468659, 67945202]
config['test_seed'] = 61302560

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/multi_mnist/multi_mnist_lenet5/alignedmtl_ub_run_2"
