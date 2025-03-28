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

from runners import SupervisedSingleTaskTrainer
config['runner'] = SupervisedSingleTaskTrainer

# dataset config
from configs.common.datasets.point_cloud_registration.train.synth_pcr_dataset_cfg import config as train_dataset_config
config.update(train_dataset_config)
from configs.common.datasets.point_cloud_registration.val.synth_pcr_dataset_cfg import config as val_dataset_config
config.update(val_dataset_config)

# model config
from configs.common.models.point_cloud_registration.geotransformer_cfg import model_config

# seeds
config['init_seed'] = 60878670
config['train_seeds'] = [87751117, 92412197, 61711150, 97099157, 64905886, 10057357, 74882964, 94355078, 13260337, 26961840, 95451222, 56918113, 91663248, 66729291, 86436114, 68233050, 87151723, 98123222, 82453262, 88801551, 58715187, 26074044, 30565041, 11691904, 27085802, 96544115, 37131059, 80761878, 40629292, 86292280, 8234759, 99077948, 80096458, 83047950, 56509977, 11563374, 28648737, 74116175, 74387353, 71986104, 55332491, 3147841, 19685920, 57867392, 35460362, 80741790, 91881319, 95064073, 8131474, 15490773, 28352696, 59863605, 59427465, 46954349, 82671366, 91103299, 39903902, 47212284, 59048122, 78820528, 22208546, 37230180, 39674883, 90924446, 11917370, 97706694, 30471360, 83178417, 60440550, 41672735, 44680822, 94716091, 18970712, 76278013, 40461161, 44581489, 18324975, 67518359, 23143687, 26651112, 99927889, 37022833, 26026894, 87374123, 56687039, 8736004, 94059041, 98358315, 78680767, 68110344, 63455455, 27194919, 63823635, 12333419, 26255596, 43644262, 7725674, 58334720, 22713579, 68588622]

# work dir
config['work_dir'] = "./logs/benchmarks/point_cloud_registration/synth_pcr_dataset/GeoTransformer_run_0"
