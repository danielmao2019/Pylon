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
from configs.common.datasets.point_cloud_registration.train.geotransformer_synth_pcr_data_cfg import data_cfg as train_data_cfg
train_data_cfg['train_dataset']['args']['overlap'] = 1.0
config.update(train_data_cfg)
from configs.common.datasets.point_cloud_registration.val.geotransformer_synth_pcr_data_cfg import data_cfg as val_data_cfg
val_data_cfg['val_dataset']['args']['overlap'] = 1.0
config.update(val_data_cfg)

# model config
from configs.common.models.point_cloud_registration.geotransformer_cfg import model_cfg
config['model'] = model_cfg

from configs.common.criteria.point_cloud_registration.geotransformer_criterion_cfg import criterion_cfg
config['criterion'] = criterion_cfg

from configs.common.metrics.point_cloud_registration.geotransformer_metric_cfg import metric_cfg
config['metric'] = metric_cfg

# seeds
config['init_seed'] = 92978070
config['train_seeds'] = [57386089, 91871188, 48479728, 44452854, 10402561, 54485068, 93910896, 49376957, 35593683, 23657400, 17641310, 52712346, 47931012, 1704496, 30342378, 87073872, 25331325, 84178425, 45998189, 90293683, 33091969, 19789322, 5336154, 50050164, 48055910, 5526276, 44983422, 75925743, 89766181, 74020106, 32938377, 93562316, 78141333, 55517733, 74413153, 32713029, 30729473, 91092589, 35820633, 70060433, 49040688, 96063213, 22982126, 97495354, 40532386, 96220313, 60801977, 47531799, 98814201, 84066815, 220602, 45817763, 44011941, 84410887, 60762113, 71838667, 95732412, 10647667, 6393238, 31475288, 80720132, 20940546, 99988482, 58303926, 93691684, 41122543, 92379921, 86771444, 61394013, 49928834, 82011615, 752127, 63623618, 77417702, 71445683, 50506629, 44501873, 7018280, 3853431, 9974228, 22225150, 81387374, 9862885, 19133905, 40184446, 93795035, 50969633, 28172968, 46943638, 34661687, 87601964, 76036618, 65508424, 84672962, 80980208, 19874101, 73535061, 97069617, 99119604, 47446562]
config['val_seeds'] = [46773800, 85336456, 87322294, 81202246, 42162042, 1900500, 89847444, 68266242, 58419945, 40328768, 63685333, 82845778, 55349761, 55275523, 94197318, 49774413, 50464473, 90028479, 91716245, 81098518, 5715361, 10988436, 69673655, 62370028, 67378705, 5134638, 29821686, 8742661, 84789393, 56827394, 32609741, 19515481, 1271950, 66191252, 98734768, 75967745, 53990988, 89327535, 86715912, 12281852, 52066076, 32252874, 20691515, 19027575, 78890225, 63850524, 53526072, 70365944, 11134999, 45176706, 81509115, 89135899, 6947028, 35527440, 12538386, 62090055, 2637181, 29336389, 16464952, 52463551, 71223213, 68427591, 26316544, 39331141, 46451571, 95809279, 26878165, 44373550, 96228124, 20095460, 54953737, 41909224, 18623221, 61292787, 87114957, 53166067, 38699136, 38739690, 83503142, 55892944, 51932871, 40156437, 85472019, 80986303, 82031261, 1476140, 63415406, 6096504, 63072151, 41422557, 29834139, 6224641, 88067242, 1407520, 74369826, 98973068, 8236281, 23389036, 77163910, 13684933]
config['test_seed'] = 1472669

# work dir
config['work_dir'] = "./logs/benchmarks/point_cloud_registration/synth_pcr_1.0/GeoTransformer_run_1"
