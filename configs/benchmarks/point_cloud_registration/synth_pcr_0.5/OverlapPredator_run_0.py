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
    'val_seeds': None,
    'test_seed': None,
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
train_data_cfg['train_dataset']['args']['overlap'] = 0.5
config.update(train_data_cfg)
from configs.common.datasets.point_cloud_registration.val.overlappredator_synth_pcr_data_cfg import data_cfg as val_data_cfg
val_data_cfg['val_dataset']['args']['overlap'] = 0.5
config.update(val_data_cfg)

# model config
from configs.common.models.point_cloud_registration.overlappredator_cfg import model_cfg
config['model'] = model_cfg

from configs.common.criteria.point_cloud_registration.overlappredator_criterion_cfg import criterion_cfg
config['criterion'] = criterion_cfg

from configs.common.metrics.point_cloud_registration.overlappredator_metric_cfg import metric_cfg
config['metric'] = metric_cfg

# seeds
config['init_seed'] = 90714354
config['train_seeds'] = [40652113, 7292667, 49752461, 58094878, 38772012, 14928894, 55947144, 41573228, 53977260, 67068629, 3302014, 10964646, 53198219, 3052054, 88894348, 25542077, 32225796, 56100954, 89751885, 17385531, 35018303, 73399031, 89401368, 47943529, 10886593, 543423, 69046726, 63106237, 54767474, 18757958, 21771589, 65141281, 59761140, 52179964, 24035428, 18561854, 5750099, 35567526, 1472295, 75945557, 29883008, 22208226, 83842412, 66164941, 64523522, 84233615, 48424332, 78650832, 14323895, 77916299, 4787777, 80525145, 80798497, 57580351, 41714035, 44176119, 6697367, 15122963, 38528595, 2489830, 49027298, 26356594, 56264362, 69966808, 63223258, 35098285, 35601345, 36785306, 59178980, 45628215, 92795134, 31215169, 35124412, 96884932, 20298365, 94698721, 29824352, 49319889, 4617260, 46071155, 13957344, 8771834, 61359026, 12274220, 57007307, 36609246, 40317499, 13663129, 89632403, 11964205, 13642141, 44721967, 61731661, 53145863, 51058837, 53478123, 24223429, 71721678, 73138618, 17254664]
config['val_seeds'] = [12155800, 751781, 56514902, 28070663, 68154191, 19623585, 75435555, 92853148, 3052040, 8298465, 66643096, 32452260, 54342491, 61833297, 99094439, 26781081, 76695850, 16083022, 75782514, 4867850, 80096084, 41642170, 37049571, 34120114, 41945294, 52834642, 65247611, 334622, 33498647, 70079996, 21685265, 17576925, 88774010, 34558954, 61646239, 48934556, 47435202, 85457518, 83695704, 47952332, 62122073, 56015264, 6270269, 74157108, 60139193, 67735414, 84796954, 62691410, 59224487, 83088912, 691681, 60340586, 87194242, 2577907, 45515140, 8169367, 80307804, 68245155, 36573570, 61155861, 20265174, 61362646, 87366973, 23802110, 25184522, 15667862, 40614898, 66709490, 441836, 32150202, 34568245, 11393555, 5948261, 10701299, 88394133, 8987451, 44791002, 65457718, 38344453, 43003806, 68749809, 338768, 5845873, 46684475, 80928598, 36768453, 27852166, 53352185, 79641045, 65937691, 62376848, 96549150, 65553929, 11677410, 34603566, 98705786, 40513553, 7606206, 60831774, 84919824]
config['test_seed'] = 46953

# work dir
config['work_dir'] = "./logs/benchmarks/point_cloud_registration/synth_pcr_0.5/OverlapPredator_run_0"
