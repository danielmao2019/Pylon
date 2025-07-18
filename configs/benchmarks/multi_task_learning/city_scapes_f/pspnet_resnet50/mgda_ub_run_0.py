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
from configs.common.optimizers.multi_task_learning.mgda_ub import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 67954733
config['train_seeds'] = [54640816, 71412476, 17108083, 23825958, 27175398, 17854379, 14294092, 79709146, 4808270, 76957181, 88216802, 45245570, 91578628, 38640168, 40778268, 82487763, 76058075, 70525968, 95253044, 3798222, 22142422, 87896881, 30954933, 44348869, 65828390, 17547774, 61802537, 55250503, 67371427, 23169582, 93516466, 5882824, 30961276, 23056568, 70294149, 48432775, 27413409, 15578136, 49554133, 57087529, 61692437, 92258264, 63233725, 14682801, 62061551, 18089438, 24986432, 68763552, 98529203, 63815025, 64006139, 50010580, 96784068, 10946911, 2860893, 48659893, 57481627, 56429117, 29155342, 2969319, 64348520, 7024274, 95709912, 478566, 84847363, 85459848, 31036581, 79947844, 49140379, 4296190, 13067274, 59101568, 57558084, 22480305, 81741376, 2001308, 87836363, 7950915, 36909163, 38117830, 43740238, 14456875, 92617549, 70414540, 81091918, 99136553, 85498646, 85452990, 26231231, 74800579, 16364935, 5189231, 4292555, 27554208, 56514717, 76636857, 7192526, 38660765, 73995896, 85693428]
config['val_seeds'] = [41401527, 52755111, 50361711, 37140241, 9599652, 69632980, 30074464, 39737671, 19555292, 14783688, 70963102, 36603688, 86690732, 32504199, 53073257, 36345399, 33413028, 16266512, 62783208, 76251857, 71151786, 75765923, 150756, 21438107, 84367959, 68037351, 43268452, 91688255, 62616322, 82706538, 96978867, 42528079, 90298253, 99153225, 12403564, 47262818, 8749555, 31558974, 22453898, 40411068, 97281346, 84577723, 73632815, 99582767, 85584812, 58763004, 61985632, 35210057, 55572952, 6642217, 8380658, 31808441, 82723007, 17177114, 90408711, 5375295, 31505009, 94305804, 26132171, 67159788, 71613320, 30250274, 97788916, 65216943, 82542305, 21954625, 59357827, 71066712, 44510563, 54401892, 17557502, 92471541, 84753616, 55362512, 95656379, 14962635, 57516424, 55493618, 55021659, 34458740, 5356134, 44806451, 92069134, 55655549, 11179469, 59738927, 20620142, 80598256, 57376555, 83494685, 77681291, 33379732, 51510905, 69328412, 3311221, 13271218, 51863869, 95583994, 45760128, 55744090]
config['test_seed'] = 55715099

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/city_scapes_f/pspnet_resnet50/mgda_ub_run_0"
