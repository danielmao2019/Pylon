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
config[0]['init_seed'] = 90714354
config[0]['train_seeds'] = [58094878, 38772012, 14928894, 55947144, 41573228, 53977260, 67068629, 3302014, 10964646, 53198219, 3052054, 88894348, 25542077, 32225796, 56100954, 89751885, 17385531, 35018303, 73399031, 89401368, 47943529, 10886593, 543423, 69046726, 63106237, 54767474, 18757958, 21771589, 65141281, 59761140, 52179964, 24035428, 18561854, 5750099, 35567526, 1472295, 75945557, 29883008, 22208226, 83842412, 66164941, 64523522, 84233615, 48424332, 78650832, 14323895, 77916299, 4787777, 80525145, 80798497, 57580351, 41714035, 44176119, 6697367, 15122963, 38528595, 2489830, 49027298, 26356594, 56264362, 69966808, 63223258, 35098285, 35601345, 36785306, 59178980, 45628215, 92795134, 31215169, 35124412, 96884932, 20298365, 94698721, 29824352, 49319889, 4617260, 46071155, 13957344, 8771834, 61359026, 12274220, 57007307, 36609246, 40317499, 13663129, 89632403, 11964205, 13642141, 44721967, 61731661, 53145863, 51058837, 53478123, 24223429, 71721678, 73138618, 17254664, 12155800, 751781, 56514902]
config[1]['init_seed'] = 40652113
config[1]['train_seeds'] = [28070663, 68154191, 19623585, 75435555, 92853148, 3052040, 8298465, 66643096, 32452260, 54342491, 61833297, 99094439, 26781081, 76695850, 16083022, 75782514, 4867850, 80096084, 41642170, 37049571, 34120114, 41945294, 52834642, 65247611, 334622, 33498647, 70079996, 21685265, 17576925, 88774010, 34558954, 61646239, 48934556, 47435202, 85457518, 83695704, 47952332, 62122073, 56015264, 6270269, 74157108, 60139193, 67735414, 84796954, 62691410, 59224487, 83088912, 691681, 60340586, 87194242, 2577907, 45515140, 8169367, 80307804, 68245155, 36573570, 61155861, 20265174, 61362646, 87366973, 23802110, 25184522, 15667862, 40614898, 66709490, 441836, 32150202, 34568245, 11393555, 5948261, 10701299, 88394133, 8987451, 44791002, 65457718, 38344453, 43003806, 68749809, 338768, 5845873, 46684475, 80928598, 36768453, 27852166, 53352185, 79641045, 65937691, 62376848, 96549150, 65553929, 11677410, 34603566, 98705786, 40513553, 7606206, 60831774, 84919824, 46953, 25548192, 44011888]
config[2]['init_seed'] = 7292667
config[2]['train_seeds'] = [9566127, 58979626, 82256604, 54752421, 26355857, 82728648, 65507348, 91359493, 45633708, 67460106, 43764436, 7492998, 49137259, 57856139, 8662884, 64855532, 56936033, 84415611, 58652081, 70748126, 77055089, 39429485, 10018585, 63229524, 9371770, 48306751, 82415971, 73038375, 69278639, 54430777, 41460162, 81439700, 61201367, 11136928, 98885792, 98972228, 39422686, 66618290, 90145464, 48635764, 39776953, 57331145, 31077044, 71762002, 98446931, 42955636, 26494257, 97666165, 1012608, 13456268, 6563475, 59981038, 85671023, 10157451, 4838895, 5186453, 61917877, 45643042, 94504283, 79963666, 73780164, 44186531, 8115847, 61348863, 39325749, 49559355, 18804647, 4496422, 71609487, 53884597, 1567306, 28778883, 63440897, 40353330, 30905048, 8413723, 24527646, 10452066, 47274685, 57615590, 29315496, 2108733, 6132486, 16079764, 93838218, 82356706, 45156911, 96273665, 29963094, 41817354, 20179013, 71990284, 52048826, 58191513, 93228272, 74480120, 14274864, 26477275, 38721962, 44317860]
config[3]['init_seed'] = 49752461
config[3]['train_seeds'] = [41355967, 83694916, 17748175, 74714013, 86020155, 11394551, 56524637, 51001762, 93076848, 79192766, 71463783, 83194887, 67960743, 4144421, 4934256, 59116793, 85859826, 62128002, 71006039, 50371783, 28670742, 98523299, 6412283, 31951009, 25256496, 87300054, 69264927, 28913813, 39993295, 80376028, 50508513, 91695640, 19055801, 79886990, 41231278, 2110806, 78622189, 15959378, 76996585, 94857203, 93837259, 20242960, 15882493, 3650406, 42023608, 72551674, 14142971, 95945372, 74316865, 39120071, 42331250, 5516019, 69168146, 60234237, 64787515, 96045178, 49095874, 58116419, 44068561, 173709, 35591427, 42211873, 42450896, 18584485, 85573400, 31701501, 87172859, 99318904, 79474984, 89142782, 58138241, 86342249, 89445665, 38723849, 29624154, 25811875, 44427, 59350185, 74745185, 53293442, 65753238, 31435854, 37064633, 29003716, 89635485, 96929622, 80412552, 10942730, 45882694, 46295304, 26746024, 84734359, 77862399, 42721076, 89478263, 74050216, 50716602, 79492701, 56673613, 11901873]

# work dir
config[0]['work_dir'] = "./logs/benchmarks/point_cloud_registration/synth_pcr_0.5/BUFFER_run_0"
config[1]['work_dir'] = "./logs/benchmarks/point_cloud_registration/synth_pcr_0.5/BUFFER_run_0"
config[2]['work_dir'] = "./logs/benchmarks/point_cloud_registration/synth_pcr_0.5/BUFFER_run_0"
config[3]['work_dir'] = "./logs/benchmarks/point_cloud_registration/synth_pcr_0.5/BUFFER_run_0"
