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
config['val_seeds'] = [76770681, 76234436, 24325365, 21437544, 97643820, 13015015, 24250460, 59113706, 81097867, 35189656, 78330969, 40907018, 84452193, 16595254, 17284517, 17560230, 33940449, 53040641, 4194965, 25489741, 4408547, 55462064, 77829984, 58917372, 55844074, 36731490, 92744402, 98726966, 30329329, 48664946, 78744627, 65569563, 93052400, 38263016, 9975074, 54822747, 79136914, 37060313, 3499072, 25627315, 4656034, 84870545, 9432200, 2275147, 64077539, 79116066, 78447539, 96836533, 40592633, 85709134, 28325590, 95185304, 56748474, 94162475, 84940680, 851759, 67193244, 48947997, 62197372, 69292725, 11235291, 90259317, 44141017, 51150562, 32121596, 36160935, 19927722, 95487586, 21191201, 22913794, 9607039, 76862249, 97302541, 88755032, 53217576, 65452579, 58587865, 37883119, 22064757, 49916543, 27053943, 71294499, 76964074, 35391345, 5608108, 58614178, 16638535, 91995892, 19631954, 5879193, 42819805, 1480220, 25306576, 11468700, 55798915, 66978380, 97172822, 71054265, 11944827, 17701464]
config['test_seed'] = 14124293

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/multi_mnist/multi_mnist_lenet5/cosreg_run_0"
