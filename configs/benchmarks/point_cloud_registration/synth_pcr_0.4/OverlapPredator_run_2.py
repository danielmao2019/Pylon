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
train_data_cfg['train_dataset']['args']['overlap'] = 0.4
config.update(train_data_cfg)
from configs.common.datasets.point_cloud_registration.val.overlappredator_synth_pcr_data_cfg import data_cfg as val_data_cfg
val_data_cfg['val_dataset']['args']['overlap'] = 0.4
config.update(val_data_cfg)

# model config
from configs.common.models.point_cloud_registration.overlappredator_cfg import model_cfg
config['model'] = model_cfg

from configs.common.criteria.point_cloud_registration.overlappredator_criterion_cfg import criterion_cfg
config['criterion'] = criterion_cfg

from configs.common.metrics.point_cloud_registration.overlappredator_metric_cfg import metric_cfg
config['metric'] = metric_cfg

# seeds
config['init_seed'] = 71045891
config['train_seeds'] = [59036167, 99249207, 10596402, 10861426, 99190617, 19456094, 28430340, 83229248, 80107656, 78225221, 36974395, 32326134, 50284604, 56239191, 35308352, 25487023, 21638590, 27180445, 72335467, 38723721, 55342457, 28666036, 60392624, 91106442, 51650349, 68262569, 12291371, 36377659, 62347564, 92897902, 68719306, 48260811, 25641916, 68769131, 81131940, 61191612, 16101653, 55506322, 22481279, 17339806, 37978403, 75207498, 2333181, 9585312, 44672508, 29419902, 60806896, 28478866, 83354226, 69582053, 98268307, 4581624, 42428647, 92132320, 91299869, 45296496, 92685420, 3053698, 84312859, 49836058, 45800143, 24599443, 82493436, 32282995, 35558307, 64016517, 77258390, 30532513, 74623597, 41224850, 18323231, 54855327, 24588489, 30830180, 51355135, 94572769, 20984726, 63753998, 15188253, 97411222, 9312368, 13681496, 79994595, 49483996, 63975187, 6972918, 47326140, 51496467, 16275634, 1175541, 45709758, 59755717, 84801789, 21317124, 97842126, 90016275, 45624481, 99507966, 77114415, 61323372]
config['val_seeds'] = [94230036, 27677175, 62219277, 82831595, 32238648, 83822458, 83286898, 6454717, 41562255, 28820295, 55554349, 10454138, 48282834, 26629392, 97574533, 17846096, 93699514, 71707863, 34495673, 61716328, 71743886, 30187647, 23651389, 89777508, 4246304, 55223611, 32023123, 34044305, 19947282, 77406705, 10520736, 75992493, 76714573, 9931824, 64611519, 17305574, 72382774, 41680626, 13882854, 24117812, 36610568, 24969860, 57214245, 48852247, 75051314, 11422522, 6943538, 72899257, 49441232, 11394007, 94781130, 5741038, 7655787, 82932202, 90032017, 9545330, 57773619, 84244299, 29236437, 36692590, 72384430, 13255677, 8784811, 81109854, 38155717, 81083544, 35349482, 76937629, 97519625, 84448300, 121426, 22357462, 40288610, 15963714, 24857315, 75804203, 8423285, 18542668, 53664131, 99102864, 15950883, 74611508, 5098118, 57467796, 76133174, 1234591, 67870736, 11677971, 15107300, 42178761, 17686643, 96750911, 50860268, 77284879, 38059317, 5553936, 80709054, 88872088, 96890964, 1103382]
config['test_seed'] = 84151357

# work dir
config['work_dir'] = "./logs/benchmarks/point_cloud_registration/synth_pcr_0.4/OverlapPredator_run_2"
