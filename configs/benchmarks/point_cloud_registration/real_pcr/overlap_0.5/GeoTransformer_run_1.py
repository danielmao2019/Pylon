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
from configs.common.datasets.point_cloud_registration.train.geotransformer_real_pcr_data_cfg import data_cfg as train_data_cfg
train_data_cfg['train_dataset']['args']['overlap'] = 0.5
config.update(train_data_cfg)
from configs.common.datasets.point_cloud_registration.val.geotransformer_real_pcr_data_cfg import data_cfg as val_data_cfg
val_data_cfg['eval_dataset']['args']['overlap'] = 0.5
config.update(val_data_cfg)

# model config
from configs.common.models.point_cloud_registration.geotransformer_cfg import model_cfg
config['model'] = model_cfg

from configs.common.criteria.point_cloud_registration.geotransformer_criterion_cfg import criterion_cfg
config['criterion'] = criterion_cfg

from configs.common.metrics.point_cloud_registration.geotransformer_metric_cfg import metric_cfg
config['metric'] = metric_cfg

# seeds
config['init_seed'] = 83637834
config['train_seeds'] = [61068132, 24318960, 33684964, 19766319, 54580573, 19806066, 57571187, 867047, 15912625, 80354006, 27839773, 89151447, 76365085, 50210073, 91343660, 84669666, 24223952, 58963544, 34927566, 56552922, 3255991, 91522567, 18990650, 71443737, 99144676, 2777633, 46630630, 97566838, 80089717, 6965331, 82915708, 62756810, 91386901, 5392226, 43775939, 19202953, 17032771, 14770080, 21273115, 3605420, 13103244, 88321387, 92318707, 7758720, 66240490, 50434971, 83226017, 27786069, 14061728, 11845488, 37478083, 48862522, 65863143, 49812057, 75343575, 32481362, 92908774, 14761581, 64481420, 19808692, 5389256, 89834622, 8219725, 84846730, 20080187, 9602704, 47078382, 7018273, 45394254, 2986504, 18257556, 33185107, 84038602, 5186670, 66565321, 85605073, 67295217, 42815885, 66698060, 16580151, 83176241, 35087889, 99928372, 21488108, 30339027, 93833617, 21987837, 66534702, 95191703, 75119005, 81700151, 71306327, 54097180, 87503466, 20115670, 81142207, 57625810, 86523618, 1895077, 87349350]

# work dir
config['work_dir'] = "./logs/benchmarks/point_cloud_registration/real_pcr/overlap_0.5/GeoTransformer_run_1"
