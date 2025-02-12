# This file is automatically generated by `./configs/benchmarks/change_detection/gen_bi_temporal.py`.
# Please do not attempt to modify manually.
import torch
import optimizers


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
    'criterion': None,
    'val_dataset': None,
    'val_dataloader': None,
    'test_dataset': None,
    'test_dataloader': None,
    'metric': None,
    # model config
    'model': None,
    # optimizer config
    'optimizer': {
        'class': optimizers.SingleTaskOptimizer,
        'args': {
            'optimizer_config': {
                'class': torch.optim.SGD,
                'args': {
                    'lr': 1.0e-03,
                    'momentum': 0.9,
                    'weight_decay': 1.0e-04,
                },
            },
        },
    },
    # scheduler config
    'scheduler': {
        'class': torch.optim.lr_scheduler.PolynomialLR,
        'args': {
            'optimizer': None,
            'total_iters': None,
            'power': 0.9,
        },
    },
}

from runners import SupervisedSingleTaskTrainer
config['runner'] = SupervisedSingleTaskTrainer

# dataset config
from configs.common.datasets.change_detection.train.oscd import config as train_dataset_config
config.update(train_dataset_config)
from configs.common.datasets.change_detection.val.oscd import config as val_dataset_config
config.update(val_dataset_config)

# model config
import models
from configs.common.models.change_detection.change_former import model_config
config['model'] = model_config
config['model']['class'] = models.change_detection.ChangeFormerV5

config['criterion']['class'] = criteria.vision_2d.ChangeFormerCriterion
# seeds
config['init_seed'] = 46583111
config['train_seeds'] = [13240630, 55940119, 19898165, 52995332, 70211650, 62356254, 75588731, 3238732, 96770085, 51546632, 24377998, 24077540, 86234555, 25053237, 68029004, 60357603, 36535775, 24809560, 65132583, 8414319, 89329011, 92127267, 15696482, 17804372, 92159622, 31220972, 14252672, 65527448, 583333, 38758437, 22797933, 25481187, 44293993, 47706071, 2388481, 77501601, 9869192, 5017114, 70559022, 72091612, 70123545, 26684764, 53320471, 36976550, 85944214, 57997141, 22923146, 22837549, 24212711, 58882402, 14934189, 65242199, 754398, 98063741, 33917695, 10739201, 34340190, 37889658, 78727581, 63553381, 20429390, 21964465, 4067414, 98716689, 88476325, 56300018, 55589355, 21269937, 70414157, 94489669, 30428295, 42290564, 50413563, 12817786, 81384564, 71196336, 71339227, 27726879, 3684809, 79324744, 61286335, 25850620, 89477863, 94525974, 19140030, 50501212, 60042569, 42771028, 63215093, 88124423, 37531686, 23596357, 18155014, 75527238, 42231792, 12003148, 56559347, 23891346, 25676661, 74068771]

# work dir
config['work_dir'] = "./logs/benchmarks/change_detection/oscd/ChangeFormerV5_run_0"
