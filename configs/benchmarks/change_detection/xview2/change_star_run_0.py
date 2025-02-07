# This file is automatically generated by `./configs/benchmarks/change_detection/gen_change_star_v1.py`.
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
                'class': schedulers.lr_lambdas.WarmupLambda,
                'args': {},
            },
        },
    },
}

from runners import SupervisedSingleTaskTrainer
config['runner'] = SupervisedSingleTaskTrainer

# dataset config
from configs.common.datasets.change_detection.train.change_star_v1_xview2 import config as train_dataset_config
config.update(train_dataset_config)
from configs.common.datasets.change_detection.val.change_star_v1_xview2 import config as val_dataset_config
config.update(val_dataset_config)

# model config
from configs.common.models.change_detection.change_star import model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config as optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 79081837
config['train_seeds'] = [3357666, 77931545, 77674551, 66065968, 52440564, 95195888, 41719557, 88574466, 81538058, 78344799, 45244149, 27635029, 8182620, 46831782, 55927427, 43399551, 56870488, 52259823, 10307464, 59278844, 49826408, 27053989, 90914132, 99560371, 67172274, 47643218, 63321423, 48378083, 96774061, 61988408, 24350848, 22886508, 20502491, 47391047, 50035688, 99508230, 44755170, 2104939, 32442363, 19843111, 66675901, 49823219, 19381253, 3919114, 23765346, 46442567, 97165712, 57779048, 71600196, 1711408, 4239317, 13518962, 16647404, 72025573, 43195661, 2620509, 37172049, 68984639, 28609655, 80077688, 1255023, 76159428, 87469140, 80866257, 60737149, 23380126, 30125543, 16555628, 32307757, 37316522, 57628423, 25620369, 87035138, 45897019, 29596461, 43751833, 69922970, 83138117, 3727748, 38206910, 14628104, 33750314, 97004626, 44248877, 37783143, 24874331, 96761093, 23806473, 37288371, 32335241, 21067141, 54301196, 76352658, 69153791, 36057486, 69630572, 14571579, 11464520, 53768241, 62575259]

# work dir
config['work_dir'] = "./logs/benchmarks/change_detection/xview2/change_star_run_0"
