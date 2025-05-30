# This file is automatically generated by `./configs/benchmarks/point_cloud_registration/gen.py`.
# Please do not attempt to modify manually.
config = {
    'runner': None,
    'work_dir': None,
    # seeds
    'seed': None,
    # dataset config
    'val_dataset': None,
    'val_dataloader': None,
    'test_dataset': None,
    'test_dataloader': None,
    'metric': None,
    # model config
    'model': None,
}

from runners import BaseEvaluator
config['runner'] = BaseEvaluator

# data config
from configs.common.datasets.point_cloud_registration.val.classic_real_pcr_data_cfg import data_cfg as eval_data_cfg
eval_data_cfg['eval_dataset']['args']['overlap'] = 0.4
config.update(eval_data_cfg)

# model config
from configs.common.models.point_cloud_registration.teaserplusplus_cfg import model_cfg
config['model'] = model_cfg

config['eval_n_jobs'] = 1

# seeds
config['seed'] = 59656834

# work dir
config['work_dir'] = "./logs/benchmarks/point_cloud_registration/real_pcr_0.4/TeaserPlusPlus_run_0"
