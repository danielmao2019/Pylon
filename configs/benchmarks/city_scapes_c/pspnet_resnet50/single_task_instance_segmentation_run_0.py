# This file is automatically generated by `./configs/benchmarks/multi_task_learning.py`.
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
from configs.common.datasets.city_scapes_c import config as dataset_config
for key in ['train_dataset', 'val_dataset', 'test_dataset']:
    dataset_config[key] = {
        'class': data.datasets.ProjectionDatasetWrapper,
        'args': {
            'dataset_config': dataset_config[key],
            'mapping': {
                'inputs': ['image'],
                'labels': ['instance_segmentation'],
                'meta_info': ['image_filepath', 'image_resolution'],
            },
        },
    }
dataset_config['criterion'] = dataset_config['criterion']['args']['criterion_configs']['instance_segmentation']
dataset_config['metric'] = dataset_config['metric']['args']['metric_configs']['instance_segmentation']
config.update(dataset_config)

# model config
from configs.common.models.city_scapes_c.pspnet_resnet50 import model_config_instance_segmentation as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers._core_ import adam_optimizer_config
config['optimizer'] = adam_optimizer_config

# seeds
config['init_seed'] = 70079096
config['train_seeds'] = [49119986, 21787158, 34998979, 83180193, 46883268, 59168776, 44395065, 10370166, 55634733, 99528040, 17808982, 70610853, 60712712, 25939329, 71921733, 68632813, 4198503, 72765446, 68328677, 44152595, 59733678, 20208404, 2608138, 5765630, 27143809, 20876430, 14788983, 22146445, 9052370, 17093818, 12038570, 20911459, 25140211, 34411491, 81237578, 42256288, 35145752, 20819524, 7663499, 32041987, 73711604, 20391818, 5498748, 54052938, 43577898, 1891744, 29550879, 4595961, 30119824, 99170451, 75164646, 37163998, 94010168, 52118908, 63586101, 19110158, 76969704, 31721141, 25914054, 50326902, 47178246, 6651740, 47704300, 11883973, 43417462, 90192927, 57886539, 12199057, 12065143, 35597264, 20584072, 30875385, 22992464, 74890352, 778578, 76258437, 96796485, 75269890, 32260021, 23523532, 97162242, 16726003, 41122963, 78076282, 10497954, 76111363, 89202931, 87618570, 72153741, 78565612, 60476265, 5855588, 98565233, 96205969, 97670601, 66544563, 40863817, 70651721, 22863703, 23061688]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_c/pspnet_resnet50/single_task_instance_segmentation_run_0"
