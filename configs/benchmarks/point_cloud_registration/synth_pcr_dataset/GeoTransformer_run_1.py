# This file is automatically generated by `./configs/benchmarks/point_cloud_registration/gen.py`.
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
                'class': torch.optim.Adam,
                'args': {
                    'params': None,
                    'lr': 1.0e-4,
                    'weight_decay': 1.0e-06,
                },
            },
        },
    },
    # scheduler config
    'scheduler': {
        'class': torch.optim.lr_scheduler.StepLR,
        'args': {
            'optimizer': None,
            'step_size': 1,
            'gamma': 0.95,
        },
    },
}

from runners import SupervisedSingleTaskTrainer
config['runner'] = SupervisedSingleTaskTrainer

# data config
from configs.common.datasets.point_cloud_registration.train.geotransformer_data_cfg import data_cfg as train_data_cfg
config.update(train_data_cfg)
from configs.common.datasets.point_cloud_registration.val.geotransformer_data_cfg import data_cfg as val_data_cfg
config.update(val_data_cfg)

# model config
from configs.common.models.point_cloud_registration.geotransformer_cfg import model_cfg
config['model'] = model_cfg

from configs.common.criteria.point_cloud_registration.geotransformer_criterion_cfg import criterion_cfg
config['criterion'] = criterion_cfg

from configs.common.metrics.point_cloud_registration.geotransformer_metric_cfg import metric_cfg
config['metric'] = metric_cfg

# seeds
config['init_seed'] = 41655509
config['train_seeds'] = [22110354, 34144158, 75263971, 68013741, 8505251, 65713471, 13450309, 20950184, 18760906, 3133071, 32273744, 63807300, 94674337, 87703538, 12784286, 91837921, 32135136, 8975525, 8107074, 85664096, 84635749, 22976287, 17639981, 15226902, 59254530, 36342651, 76514861, 97112902, 7178698, 88740442, 13696338, 31697122, 8858044, 49678125, 44011666, 90499136, 12000023, 97296442, 91629950, 9664124, 61179051, 43152406, 10599689, 91282070, 28626160, 15022683, 27524179, 20547396, 6791758, 40272739, 57822254, 97462257, 4487518, 65817821, 10486386, 47048967, 19714979, 42731661, 6219167, 87486507, 88465441, 11237589, 75419186, 63006805, 2213836, 66332774, 37895689, 24361961, 32377854, 46735065, 6770625, 16804553, 97817204, 87812203, 72475158, 16469225, 1466490, 64564175, 59907603, 59156425, 45342386, 70443551, 62138582, 11963983, 87075833, 48588714, 62657367, 92599566, 85018964, 87335239, 80805775, 23651360, 37762985, 75322035, 45650932, 43770825, 95880451, 17382137, 90905482, 6543286]

# work dir
config['work_dir'] = "./logs/benchmarks/point_cloud_registration/synth_pcr_dataset/GeoTransformer_run_1"
