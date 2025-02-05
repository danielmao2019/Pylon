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

from runners import SupervisedMultiTaskTrainer
config['runner'] = SupervisedMultiTaskTrainer

# dataset config
from configs.common.datasets.multi_task_learning.multi_mnist import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.multi_mnist.multi_mnist_lenet5 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.cosreg import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 94646818
config['train_seeds'] = [80581783, 9019961, 61759741, 36643733, 60092032, 67755282, 43198171, 28734359, 2790642, 15734643, 3680367, 75988606, 72248647, 15656048, 41115455, 9867761, 4207068, 7723870, 36865122, 92429924, 5257592, 39199130, 81632552, 30094488, 53725962, 10093743, 63009286, 8445961, 68872069, 76385612, 49773658, 40861857, 31971327, 88808750, 40438119, 28902908, 50312118, 11237164, 5405532, 37378100, 91297081, 54566045, 61353507, 96925809, 69896678, 21902533, 13045036, 78183458, 69016111, 85467178, 73039952, 75283646, 97699274, 65569697, 36583835, 86247305, 67227888, 12466203, 50626109, 37125249, 34490597, 84746692, 45996235, 49645284, 49882527, 9823026, 18871414, 79729053, 61199007, 57193731, 40360707, 22058685, 98636029, 1716535, 50572435, 6354469, 76503908, 34634562, 25196233, 93858484, 52423490, 50911828, 67754981, 6265334, 28407060, 46571038, 22940526, 76771551, 5724159, 76931939, 55683754, 97676835, 64381554, 19817171, 63659637, 40500099, 33910318, 89154509, 48559296, 60458648]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/multi_mnist/multi_mnist_lenet5/cosreg_run_0"
