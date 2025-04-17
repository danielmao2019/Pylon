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
            'step_size': 1000,
            'gamma': 0.95,
        },
    },
}

from runners import SupervisedSingleTaskTrainer
config['runner'] = SupervisedSingleTaskTrainer

# data config
from configs.common.datasets.point_cloud_registration.train.overlappredator_synth_pcr_data_cfg import data_cfg as train_data_cfg
train_data_cfg['train_dataset']['args']['overlap'] = 1.0
config.update(train_data_cfg)
from configs.common.datasets.point_cloud_registration.val.overlappredator_synth_pcr_data_cfg import data_cfg as val_data_cfg
val_data_cfg['val_dataset']['args']['overlap'] = 1.0
config.update(val_data_cfg)

# model config
from configs.common.models.point_cloud_registration.overlappredator_cfg import model_cfg
config['model'] = model_cfg

from configs.common.criteria.point_cloud_registration.overlappredator_criterion_cfg import criterion_cfg
config['criterion'] = criterion_cfg

from configs.common.metrics.point_cloud_registration.overlappredator_metric_cfg import metric_cfg
config['metric'] = metric_cfg

# seeds
config['init_seed'] = 24160206
config['train_seeds'] = [15400517, 56075889, 89948040, 78502627, 54651685, 21329377, 71477441, 28842558, 20862214, 75419081, 18239896, 12237205, 14253217, 13173250, 46919013, 80734184, 47044586, 20940616, 81392669, 55674079, 90699943, 68066031, 458063, 99388858, 83974322, 54533303, 69116491, 34983326, 90236337, 31095876, 93673213, 97051309, 12100440, 5941972, 15182591, 79898894, 28551889, 19098392, 82397292, 76661768, 4616819, 26448736, 13317781, 11457906, 78893323, 91307808, 30512255, 92020434, 85548393, 25673542, 39426588, 40005442, 69080396, 29610749, 65207273, 91457506, 23985623, 48994401, 67219052, 58868407, 41039504, 89471351, 60161736, 73807348, 3945975, 49311648, 28599084, 45896408, 58260724, 85573015, 66325117, 18375796, 40496685, 28941580, 35566473, 38068476, 38365210, 59603614, 5036281, 51033574, 18263641, 14360117, 42284310, 43270029, 83998145, 6651560, 87033558, 237587, 79086335, 94432455, 96845020, 11839868, 29405524, 81233968, 53727322, 23993394, 51189716, 99917528, 66841105, 78471843]

# work dir
config['work_dir'] = "./logs/benchmarks/point_cloud_registration/synth_pcr/overlap_1.0/OverlapPredator_run_1"
