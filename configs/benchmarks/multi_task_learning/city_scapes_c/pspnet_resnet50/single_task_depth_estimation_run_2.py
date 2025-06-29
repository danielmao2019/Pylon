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

from runners import SupervisedSingleTaskTrainer
config['runner'] = SupervisedSingleTaskTrainer

# dataset config
import data
from configs.common.datasets.multi_task_learning.city_scapes_c import config as dataset_config
for key in ['train_dataset', 'val_dataset', 'test_dataset']:
    dataset_config[key] = {
        'class': data.datasets.ProjectionDatasetWrapper,
        'args': {
            'dataset_config': dataset_config[key],
            'mapping': {
                'inputs': ['image'],
                'labels': ['depth_estimation'],
                'meta_info': ['image_filepath', 'image_resolution'],
            },
        },
    }
dataset_config['criterion'] = dataset_config['criterion']['args']['criterion_configs']['depth_estimation']
dataset_config['metric'] = dataset_config['metric']['args']['metric_configs']['depth_estimation']
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.city_scapes_c.pspnet_resnet50 import model_config_depth_estimation as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config as optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 10020618
config['train_seeds'] = [13329446, 36091132, 1756576, 5956642, 14140996, 13988000, 59093511, 77888855, 81541490, 17303792, 14750074, 18926576, 36137347, 41793544, 10485976, 79269933, 4629468, 20959085, 5483227, 43994702, 75222078, 38261, 80403409, 71867613, 53076067, 17743929, 12412689, 16460016, 23007791, 35899805, 82336882, 38144853, 96127744, 84033265, 99297435, 86247750, 38290944, 6264107, 22506960, 43211098, 77608713, 12987267, 33129389, 5847489, 48235883, 92101912, 66273844, 8873295, 21972489, 89351485, 99240207, 72756923, 50028582, 40677273, 73900946, 18277746, 4873431, 59397673, 49268232, 38292628, 25663624, 84577935, 46254125, 21723040, 82208354, 27840700, 24512613, 43892853, 36511760, 78068302, 54977209, 33456549, 14778572, 39243655, 84169816, 95002200, 18498558, 62754968, 51027209, 34403441, 239985, 24000288, 24948987, 54249148, 66507692, 59784912, 1124297, 36304998, 32179973, 54381397, 47185219, 11529925, 25904668, 97591257, 68740876, 62641730, 97586959, 70226999, 85995145, 60360005]
config['val_seeds'] = [25224096, 17604406, 37200422, 19640647, 56922017, 18636582, 84267836, 89978303, 23162521, 43276189, 35507875, 95623204, 21763765, 26236261, 50105617, 96678936, 50224427, 2191223, 57060206, 55416506, 51868175, 20305287, 35526668, 597626, 22303565, 58390789, 75537119, 66017091, 57927814, 3353973, 49705980, 20468597, 91530007, 97826928, 27935882, 13190648, 47299956, 87140874, 53931170, 40756138, 24834384, 81890317, 52244204, 69037885, 57840031, 47561699, 58241069, 78237069, 16644860, 2806926, 61102585, 37491209, 38185265, 2474495, 65840598, 35918986, 397704, 71273267, 7198411, 68256630, 52239497, 65437547, 13991227, 94809942, 34576864, 32034834, 54261659, 99653280, 74907271, 43416902, 3698502, 50001536, 64982730, 12160896, 18849449, 55143303, 59351668, 73841611, 20323086, 73729280, 75972657, 3766578, 24069130, 9925043, 9369451, 95901900, 30834934, 98215437, 99102129, 45696010, 74701425, 54449118, 88347584, 5079471, 57034624, 12188306, 46622242, 57117894, 87781766, 99891317]
config['test_seed'] = 17890221

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/city_scapes_c/pspnet_resnet50/single_task_depth_estimation_run_2"
