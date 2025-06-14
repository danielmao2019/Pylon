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
from configs.common.optimizers.multi_task_learning.mgda_ub import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 43440641
config['train_seeds'] = [66832673, 20779490, 85764619, 31518714, 59516631, 54891963, 17209497, 7878382, 85243956, 59367530, 3554733, 83610871, 11176266, 6025915, 86323830, 74546240, 57119602, 50884417, 9733685, 10094931, 55646025, 15033844, 3934427, 58636649, 39315605, 81004874, 27789871, 45622239, 32182975, 63897882, 56625106, 97695108, 21284473, 64848270, 24551332, 65950791, 92822105, 12993233, 71032990, 71549937, 87478896, 6212148, 83018514, 82466867, 22974394, 7138274, 52572243, 35923588, 37990503, 95565872, 58712768, 43910198, 78012773, 71419105, 2336979, 91731821, 90684782, 6438231, 94628545, 86174418, 86851001, 653901, 78090152, 31884394, 62350195, 16240497, 83244251, 58570110, 7557080, 63849961, 42131563, 58391543, 52541858, 85304123, 357977, 28247512, 18498714, 29901632, 37998597, 9079483, 81173106, 65890000, 99799087, 60120934, 69785667, 1040518, 10514463, 97675534, 38801873, 65737230, 89973898, 98114825, 98463440, 20260008, 91772236, 56914343, 64904832, 13818409, 66100067, 4299251]
config['val_seeds'] = [11540790, 40183002, 67956294, 69678147, 34488503, 62214940, 92466371, 63516120, 3043243, 9309637, 75522504, 75868677, 47055623, 28880876, 37054624, 84032885, 26514893, 75456526, 95398080, 36967081, 62267301, 46719027, 46175960, 30797943, 47696184, 96081475, 21993392, 92463119, 91688703, 55072835, 98911558, 57005079, 30706623, 99230117, 28713928, 67107164, 11152323, 62161574, 10694160, 76329334, 90273418, 55320169, 79628746, 70325751, 84105481, 80247123, 93334410, 91545522, 89965655, 80073163, 45936063, 70191511, 6466566, 79852240, 14728368, 22974083, 46963309, 71697373, 32661553, 28987070, 4558545, 52477545, 38713196, 85730864, 67897156, 93104567, 8361463, 71282310, 65363515, 30914616, 27300884, 75277483, 83469100, 61507994, 29949627, 44961693, 60790697, 28350109, 32432149, 2499841, 37866395, 48143496, 61346891, 68020148, 38197758, 34064685, 75427012, 23149886, 10798109, 60049578, 39064522, 15954114, 57614259, 59333127, 35108359, 483591, 42411994, 77924823, 70439714, 85413829]
config['test_seed'] = 8524619

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/nyu_v2_f/pspnet_resnet50/mgda_ub_run_0"
