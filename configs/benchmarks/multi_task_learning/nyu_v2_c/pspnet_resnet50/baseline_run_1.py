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
from configs.common.datasets.multi_task_learning.nyu_v2_c import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.nyu_v2_c.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.baseline import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 65447448
config['train_seeds'] = [51010300, 99774920, 76745620, 70849317, 98084520, 58451399, 4654078, 48480202, 51746214, 63349512, 28348952, 51496669, 11417143, 59452279, 39450358, 98582661, 44903008, 71010433, 96463772, 37215027, 54656783, 71679038, 41197705, 1062619, 96604786, 77162649, 11462780, 55836563, 93107557, 42761860, 73763978, 36234704, 4476239, 3988699, 9122587, 43631259, 11376795, 65863711, 44229691, 91292875, 44552885, 87090927, 40075291, 4454616, 67569109, 66088966, 40069230, 4886813, 44923044, 29593609, 51305969, 53525572, 5046670, 36765933, 48568649, 14723092, 38613850, 92761406, 25020694, 117248, 3169258, 20264547, 10525775, 73052401, 53415158, 43073939, 9831694, 52437737, 87359726, 50961832, 56175980, 88313096, 15518729, 1159695, 50885580, 12531935, 47088899, 57975110, 54322155, 28185670, 43180721, 16381033, 59793892, 51579271, 72527066, 60575726, 85924811, 72576570, 75089324, 92890028, 52863082, 38307503, 5527236, 86217565, 21239974, 26444484, 48463375, 26138207, 65468915, 90636414]
config['val_seeds'] = [20467367, 11863616, 58289958, 26633682, 1573230, 15556945, 71473880, 91381987, 43991304, 71214031, 87984369, 62699285, 5427410, 49766200, 15593875, 28092472, 88136060, 50833203, 41272134, 72567515, 97014172, 48316529, 2406419, 81997175, 99537501, 40332249, 75905673, 35099422, 42044854, 77319390, 79176224, 73117561, 54753888, 76151520, 79852595, 19431470, 57493311, 63960612, 92650543, 9722031, 51212713, 75308260, 25670563, 81755660, 14592827, 87314443, 50186275, 5088770, 67054127, 4431643, 98907041, 76798041, 5808491, 79265045, 11602097, 22586499, 54090937, 15143257, 54563900, 62460958, 54796805, 58089752, 24410268, 60526980, 39703766, 46003439, 77263366, 74758464, 75810370, 18776682, 14055494, 80069583, 16952146, 99642727, 667945, 61044048, 51968644, 80595920, 10662683, 99879202, 35970447, 76192525, 91918021, 497558, 52900897, 3281700, 46296393, 23157388, 68544098, 14161504, 39076224, 43635327, 62260283, 25682182, 31337626, 34985453, 93894162, 9375408, 85617086, 12130704]
config['test_seed'] = 86411680

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/nyu_v2_c/pspnet_resnet50/baseline_run_1"
