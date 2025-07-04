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

from runners import SupervisedMultiTaskTrainer
config['runner'] = SupervisedMultiTaskTrainer

# dataset config
from configs.common.datasets.multi_task_learning.city_scapes_f import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.city_scapes_f.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.cagrad import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 2533083
config['train_seeds'] = [11082401, 53700117, 79342544, 38098596, 54266470, 43162544, 95847712, 8803272, 52439863, 99434626, 13699751, 75743263, 88284719, 14108918, 30252253, 53325108, 93588444, 97039414, 54518519, 10476183, 80781523, 56619199, 17028877, 48154950, 69051016, 2146712, 40558423, 79094929, 99672887, 26582009, 54930482, 62621532, 1650542, 9884698, 10463774, 94837579, 13387948, 78667065, 42649551, 50633836, 58749990, 55663139, 78693209, 77448888, 16449464, 31117962, 45445909, 82084407, 79771292, 82445809, 72580943, 5511114, 33827447, 24645872, 85378176, 3459465, 65187649, 82156250, 39520771, 60266281, 21141772, 90349183, 88383349, 39131663, 26559453, 31773063, 81817792, 50311308, 44806723, 66982242, 15272252, 31089805, 99472149, 71257440, 84593745, 89942972, 45819125, 69511920, 30075469, 58485171, 16091480, 82170639, 69793697, 68968590, 8218138, 30758037, 52774169, 73601225, 8676531, 50176425, 77391074, 75381201, 66025721, 95370235, 5694736, 21964484, 12933660, 425297, 68467386, 41734937]
config['val_seeds'] = [71281751, 25389184, 29197392, 95672112, 8890313, 18934461, 64477994, 14131139, 89240727, 45531188, 13938570, 68789945, 47364718, 92685418, 83356341, 51063325, 47113084, 14475581, 16904357, 82533825, 17268879, 87541455, 66205134, 96692096, 68432298, 60648248, 84005759, 46045128, 62553487, 15086353, 24922308, 37058966, 4044212, 80055422, 94875270, 56223831, 11316199, 59184264, 5276647, 72345908, 12974570, 73608332, 98606022, 9352077, 62571112, 17835873, 10878522, 24247235, 36119397, 69129963, 22349256, 39385691, 95815738, 81569951, 43125564, 73058154, 34446899, 83579911, 79528663, 97333626, 44290023, 70959957, 36780615, 17998512, 8905358, 99329901, 82361429, 21134389, 71220223, 29963844, 86517418, 33246892, 89790456, 51437652, 95396276, 41005749, 51151319, 8524798, 15544862, 26853151, 67952697, 37499141, 53362273, 25414814, 78858445, 52481507, 20877046, 32409668, 98286504, 19210189, 39647565, 35602317, 11135940, 49221201, 32393609, 88684840, 61291538, 10838903, 20023537, 29953932]
config['test_seed'] = 85815398

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/city_scapes_f/pspnet_resnet50/cagrad_run_1"
