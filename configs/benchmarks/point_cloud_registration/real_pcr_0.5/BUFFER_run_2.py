# This file is automatically generated by `./configs/benchmarks/point_cloud_registration/gen.py`.
# Please do not attempt to modify manually.
import torch
import criteria
import metrics
import optimizers
import runners.pcr_trainers

import copy
from configs.common.datasets.point_cloud_registration.train.buffer_data_cfg import get_transforms_cfg
from configs.common.datasets.point_cloud_registration.train.buffer_data_cfg import data_cfg as train_data_cfg
from configs.common.datasets.point_cloud_registration.val.buffer_data_cfg import data_cfg as val_data_cfg
from configs.common.models.point_cloud_registration.buffer_cfg import model_cfg

optimizer_cfg = {
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
}

scheduler_cfg = {
    'class': torch.optim.lr_scheduler.StepLR,
    'args': {
        'optimizer': None,
        'step_size': 1000,
        'gamma': 0.95,
    },
}

config = [
{
    'stage': 'Ref',
    'runner': runners.pcr_trainers.BufferTrainer,
    'work_dir': None,
    'epochs': 100,
    # seeds
    'init_seed': None,
    'train_seeds': None,
    # dataset config
    'train_dataset': copy.deepcopy(train_data_cfg['train_dataset']),
    'train_dataloader': copy.deepcopy(train_data_cfg['train_dataloader']),
    'criterion': {
        'class': criteria.vision_3d.point_cloud_registration.BUFFER_RefStageCriterion,
        'args': {},
    },
    'val_dataset': copy.deepcopy(val_data_cfg['val_dataset']),
    'val_dataloader': copy.deepcopy(val_data_cfg['val_dataloader']),
    'test_dataset': None,
    'test_dataloader': None,
    'metric': {
        'class': metrics.vision_3d.point_cloud_registration.BUFFER_RefStageMetric,
        'args': {},
    },
    # model config
    'model': copy.deepcopy(model_cfg),
    # optimizer config
    'optimizer': copy.deepcopy(optimizer_cfg),
    # scheduler config
    'scheduler': copy.deepcopy(scheduler_cfg),
},
{
    'stage': 'Desc',
    'runner': runners.pcr_trainers.BufferTrainer,
    'work_dir': None,
    'epochs': 100,
    # seeds
    'init_seed': None,
    'train_seeds': None,
    # dataset config
    'train_dataset': copy.deepcopy(train_data_cfg['train_dataset']),
    'train_dataloader': copy.deepcopy(train_data_cfg['train_dataloader']),
    'criterion': {
        'class': criteria.vision_3d.point_cloud_registration.BUFFER_DescStageCriterion,
        'args': {},
    },
    'val_dataset': copy.deepcopy(val_data_cfg['val_dataset']),
    'val_dataloader': copy.deepcopy(val_data_cfg['val_dataloader']),
    'test_dataset': None,
    'test_dataloader': None,
    'metric': {
        'class': metrics.vision_3d.point_cloud_registration.BUFFER_DescStageMetric,
        'args': {},
    },
    # model config
    'model': copy.deepcopy(model_cfg),
    # optimizer config
    'optimizer': copy.deepcopy(optimizer_cfg),
    # scheduler config
    'scheduler': copy.deepcopy(scheduler_cfg),
},
{
    'stage': 'Keypt',
    'runner': runners.pcr_trainers.BufferTrainer,
    'work_dir': None,
    'epochs': 100,
    # seeds
    'init_seed': None,
    'train_seeds': None,
    # dataset config
    'train_dataset': copy.deepcopy(train_data_cfg['train_dataset']),
    'train_dataloader': copy.deepcopy(train_data_cfg['train_dataloader']),
    'criterion': {
        'class': criteria.vision_3d.point_cloud_registration.BUFFER_KeyptStageCriterion,
        'args': {},
    },
    'val_dataset': copy.deepcopy(val_data_cfg['val_dataset']),
    'val_dataloader': copy.deepcopy(val_data_cfg['val_dataloader']),
    'test_dataset': None,
    'test_dataloader': None,
    'metric': {
        'class': metrics.vision_3d.point_cloud_registration.BUFFER_KeyptStageMetric,
        'args': {},
    },
    # model config
    'model': copy.deepcopy(model_cfg),
    # optimizer config
    'optimizer': copy.deepcopy(optimizer_cfg),
    # scheduler config
    'scheduler': copy.deepcopy(scheduler_cfg),
},
{
    'stage': 'Inlier',
    'runner': runners.pcr_trainers.BufferTrainer,
    'work_dir': None,
    'epochs': 100,
    # seeds
    'init_seed': None,
    'train_seeds': None,
    # dataset config
    'train_dataset': copy.deepcopy(train_data_cfg['train_dataset']),
    'train_dataloader': copy.deepcopy(train_data_cfg['train_dataloader']),
    'criterion': {
        'class': criteria.vision_3d.point_cloud_registration.BUFFER_InlierStageCriterion,
        'args': {},
    },
    'val_dataset': copy.deepcopy(val_data_cfg['val_dataset']),
    'val_dataloader': copy.deepcopy(val_data_cfg['val_dataloader']),
    'test_dataset': None,
    'test_dataloader': None,
    'metric': {
        'class': metrics.vision_3d.point_cloud_registration.BUFFER_InlierStageMetric,
        'args': {},
    },
    # model config
    'model': copy.deepcopy(model_cfg),
    # optimizer config
    'optimizer': copy.deepcopy(optimizer_cfg),
    # scheduler config
    'scheduler': copy.deepcopy(scheduler_cfg),
},
]

config[0]['train_dataset']['args']['transforms_cfg'] = get_transforms_cfg('Euler', 3)
config[1]['train_dataset']['args']['transforms_cfg'] = get_transforms_cfg('Euler', 1)
config[2]['train_dataset']['args']['transforms_cfg'] = get_transforms_cfg('Euler', 1)
config[3]['train_dataset']['args']['transforms_cfg'] = get_transforms_cfg('Euler', 1)

config[0]['val_dataset']['args']['transforms_cfg'] = get_transforms_cfg('Euler', 3)
config[1]['val_dataset']['args']['transforms_cfg'] = get_transforms_cfg('Euler', 1)
config[2]['val_dataset']['args']['transforms_cfg'] = get_transforms_cfg('Euler', 1)
config[3]['val_dataset']['args']['transforms_cfg'] = get_transforms_cfg('Euler', 1)

config[0]['model']['args']['config']['stage'] = 'Ref'
config[1]['model']['args']['config']['stage'] = 'Desc'
config[2]['model']['args']['config']['stage'] = 'Keypt'
config[3]['model']['args']['config']['stage'] = 'Inlier'

# seeds
config[0]['init_seed'] = 18250117
config[0]['train_seeds'] = [97636575, 6600893, 25256834, 88160551, 32971409, 6170598, 16057766, 43140656, 43522812, 9478223, 73475725, 94850068, 10558901, 49056875, 80394633, 66397453, 55203237, 61117287, 97735024, 19500588, 76381290, 45190796, 31828909, 35576195, 12328713, 86163743, 48968728, 21220433, 13759145, 29673585, 33461172, 24965425, 62463345, 88802112, 90996601, 30679642, 98787406, 36898027, 41806734, 55512412, 39600619, 72496012, 83191664, 45931718, 73428143, 82472015, 6057345, 93451565, 85133042, 43976447, 90122870, 27415985, 29681080, 31440430, 27204213, 30672127, 89372548, 7686079, 63686423, 87902901, 58080792, 71578588, 35022439, 6341716, 34229406, 1139105, 70626484, 62133388, 76291384, 20271496, 35523510, 54249197, 75258033, 71496990, 43902338, 27015621, 35677652, 95829635, 47925275, 11047736, 21142854, 45595428, 73776569, 61184128, 23493237, 17064001, 23632958, 76760291, 27994429, 5802664, 76543782, 89209677, 42957496, 5723808, 91251176, 65571161, 57451429, 98262866, 36661610, 77461331]
config[1]['init_seed'] = 36065661
config[1]['train_seeds'] = [62175685, 56026280, 59486594, 35848065, 98561697, 70730564, 9856000, 85933874, 5199541, 82798173, 99837376, 2352645, 10759996, 35857742, 88192848, 73541583, 64082215, 91529246, 27123619, 19582729, 52690502, 27383051, 77498853, 7235717, 77612045, 68207437, 76063268, 33637485, 92257399, 93300742, 87121521, 34444725, 70117276, 60886533, 88793528, 57689369, 97411716, 58885060, 7210828, 14400320, 55396395, 61101738, 81743779, 6183148, 60618621, 81493556, 77918594, 3666148, 48911994, 74719212, 74272916, 82950955, 89757435, 48406986, 33515441, 34447871, 57846402, 84674734, 41168273, 18558920, 45630260, 15528706, 55753661, 12110711, 98250654, 15627935, 1585886, 99396532, 58756753, 27160337, 25375379, 30599822, 1906702, 2962798, 30040888, 19069071, 22131606, 47459552, 40966220, 41998878, 81358179, 21087917, 64030627, 47093404, 54858975, 32300617, 40863632, 10670501, 53161229, 29456041, 63520238, 81606193, 48797627, 86430576, 78536978, 77563520, 61276624, 7840945, 82605859, 20470083]
config[2]['init_seed'] = 83994145
config[2]['train_seeds'] = [18337289, 72190192, 79214361, 24736368, 22041177, 74344990, 89281345, 91669490, 74236026, 57526119, 81741488, 6266861, 5851665, 1306108, 68959939, 49762234, 85852084, 49286945, 52467633, 67817151, 91184083, 84194564, 96096430, 36682469, 93587593, 29176956, 25745077, 66554046, 75972000, 95007094, 62423728, 85665650, 67145006, 44544917, 90290003, 73886837, 64090479, 30993754, 10934798, 57119450, 94806358, 64674661, 42464512, 71699701, 7528557, 26128845, 61763659, 93474150, 98293743, 49827231, 67188412, 92174990, 87263319, 69134609, 69763995, 11627676, 9306013, 12804267, 98542424, 64282548, 82873736, 61904074, 96928650, 23543051, 9555971, 20884717, 30741961, 4079993, 84489, 19542886, 80011958, 45963098, 70499115, 65149111, 54174821, 68188709, 70656467, 68645686, 61895874, 30509167, 90339484, 60105164, 3442369, 87395161, 63346039, 83756444, 62767125, 70101435, 8181705, 30267396, 25987066, 4565003, 55380300, 32434755, 16720461, 57343450, 25739877, 34094089, 39662531, 54156705]
config[3]['init_seed'] = 22177182
config[3]['train_seeds'] = [55697829, 61297753, 18654513, 3419141, 86057997, 24198945, 64001521, 13273428, 85922585, 51581477, 80195484, 16969000, 24603124, 543634, 39085248, 7217444, 40842546, 52511785, 82854530, 63435278, 13999685, 39818767, 48201177, 68127952, 83698304, 25111314, 28998704, 60577558, 84488341, 84440781, 96751431, 28342351, 39571152, 70984796, 4726684, 56858133, 95655023, 31151021, 44099873, 1410660, 50291741, 94933895, 90811789, 23525677, 74367829, 51334820, 48854261, 62862793, 81329859, 64178945, 42303931, 21031684, 33273779, 68671022, 75402231, 47877445, 69542894, 96734168, 4639862, 85038515, 88483322, 91088611, 52564821, 28110050, 94178139, 38494944, 23293376, 93114164, 75175271, 74946790, 44923032, 89621311, 22203921, 4220345, 94554659, 86674084, 58488721, 51102487, 75116663, 47706582, 54388725, 23477995, 44078299, 49907606, 55450915, 57945232, 52471399, 87097565, 60859582, 17923446, 75395645, 71483511, 13027228, 90653749, 80935722, 40652323, 4612909, 14098653, 91024289, 21217547]

# work dir
config[0]['work_dir'] = "./logs/benchmarks/point_cloud_registration/real_pcr_0.5/BUFFER_run_2"
config[1]['work_dir'] = "./logs/benchmarks/point_cloud_registration/real_pcr_0.5/BUFFER_run_2"
config[2]['work_dir'] = "./logs/benchmarks/point_cloud_registration/real_pcr_0.5/BUFFER_run_2"
config[3]['work_dir'] = "./logs/benchmarks/point_cloud_registration/real_pcr_0.5/BUFFER_run_2"
