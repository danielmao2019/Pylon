# This file is automatically generated by `./configs/benchmarks/change_detection/gen_i3pe.py`.
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

from runners import MultiValDatasetTrainer
config['runner'] = MultiValDatasetTrainer

# dataset config
from configs.common.datasets.change_detection.train.i3pe_sysu_cd import config as train_dataset_config
config.update(train_dataset_config)
from configs.common.datasets.change_detection.val.all_bi_temporal import config as val_dataset_config
config.update(val_dataset_config)

# model config
from configs.common.models.change_detection.i3pe_model import model_config
config['model'] = model_config

# seeds
config['init_seed'] = 31211687
config['train_seeds'] = [5051043, 73617491, 32750541, 45473122, 14435062, 58097101, 70457671, 92686240, 94001752, 37344283, 44716394, 98739538, 31194775, 41428762, 52462947, 94824652, 35126637, 6620602, 11534597, 70641828, 34067123, 55133978, 40360820, 94560779, 60498069, 56294531, 56203422, 21669270, 66931749, 34806760, 9883907, 36968768, 89862754, 94335150, 22184232, 65455653, 69412653, 23311373, 64833698, 20003873, 54143767, 72908644, 41871819, 39635147, 48971440, 91524154, 55260209, 84378210, 30526236, 30308961, 48172992, 78782391, 69427370, 91424928, 21667818, 16059307, 96567962, 1641787, 53787839, 52182053, 4418780, 91615842, 18301466, 33931089, 56594513, 39144686, 36704425, 6107567, 7057272, 5645640, 4880461, 82195314, 40370474, 18766612, 41045078, 63469723, 92409265, 97702046, 75082750, 24914196, 41828991, 67019436, 36053917, 40238079, 49597256, 93344363, 50630065, 53966917, 5742785, 83531441, 12167688, 85977561, 18295463, 90509917, 99926260, 62511922, 24216926, 84356605, 61141014, 14266126]

# work dir
config['work_dir'] = "./logs/benchmarks/change_detection/i3pe/sysu_cd_run_0"
