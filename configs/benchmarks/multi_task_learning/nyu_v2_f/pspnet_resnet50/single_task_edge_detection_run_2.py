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
                'class': schedulers.WarmupLambda,
                'args': {},
            },
        },
    },
}

from runners import SupervisedSingleTaskTrainer
config['runner'] = SupervisedSingleTaskTrainer

# dataset config
import data
from configs.common.datasets.multi_task_learning.nyu_v2_f import config as dataset_config
for key in ['train_dataset', 'val_dataset', 'test_dataset']:
    dataset_config[key] = {
        'class': data.datasets.ProjectionDatasetWrapper,
        'args': {
            'dataset_config': dataset_config[key],
            'mapping': {
                'inputs': ['image'],
                'labels': ['edge_detection'],
                'meta_info': ['image_filepath', 'image_resolution'],
            },
        },
    }
dataset_config['criterion'] = dataset_config['criterion']['args']['criterion_configs']['edge_detection']
dataset_config['metric'] = dataset_config['metric']['args']['metric_configs']['edge_detection']
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.nyu_v2_f.pspnet_resnet50 import model_config_edge_detection as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config as optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 90234376
config['train_seeds'] = [52703335, 68736201, 49553670, 80748433, 22438786, 86786846, 41242601, 93762834, 74048774, 31441455, 7160637, 87455183, 84761052, 26634561, 99613932, 93228317, 40681293, 8427377, 41583667, 57111861, 26622481, 79884498, 12307592, 97629892, 93217144, 83083926, 74072034, 75148139, 3748948, 11125065, 31359028, 81821445, 33705123, 27057528, 98033108, 66766501, 75121730, 10459731, 30805182, 56942496, 67075357, 88186242, 40258601, 67777701, 3401915, 16779952, 54222800, 89755636, 80149325, 75773706, 52160224, 53199322, 16877630, 16543798, 63009739, 49480597, 99978157, 33996206, 42373002, 92537980, 62264335, 16473556, 48166658, 26220878, 37068461, 95326497, 45079756, 1440125, 43529857, 95543044, 64287369, 14476782, 71281917, 4027054, 26793403, 69297233, 11503527, 64968131, 92004099, 51598737, 21322638, 6624619, 96097639, 51043401, 10467624, 31177636, 16973350, 61749503, 59190328, 29833080, 93038293, 59202165, 89199403, 3787012, 54447249, 98577322, 81505759, 76162897, 11978842, 25604036]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/nyu_v2_f/pspnet_resnet50/single_task_edge_detection_run_2"
