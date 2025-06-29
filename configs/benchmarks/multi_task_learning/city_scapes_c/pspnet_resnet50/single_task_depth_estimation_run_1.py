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

from runners import SupervisedSingleTaskTrainer
config['runner'] = SupervisedSingleTaskTrainer

# dataset config
import data
from configs.common.datasets.multi_task_learning.city_scapes_c import config as dataset_config
for key in ['train_dataset', 'val_dataset', 'test_dataset']:
    dataset_config[key] = {
        'class': data.datasets.ProjectionDatasetWrapper,
        'args': {
            'dataset_config': dataset_config[key],
            'mapping': {
                'inputs': ['image'],
                'labels': ['depth_estimation'],
                'meta_info': ['image_filepath', 'image_resolution'],
            },
        },
    }
dataset_config['criterion'] = dataset_config['criterion']['args']['criterion_configs']['depth_estimation']
dataset_config['metric'] = dataset_config['metric']['args']['metric_configs']['depth_estimation']
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.city_scapes_c.pspnet_resnet50 import model_config_depth_estimation as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config as optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 77591364
config['train_seeds'] = [44360212, 56422211, 15365249, 73112475, 71416866, 4793896, 36912577, 73813468, 88186393, 6716551, 33074288, 45691591, 14434169, 77398732, 37653135, 80016119, 58043519, 48786287, 95504044, 91756315, 87337707, 10592991, 63960939, 63907091, 67197958, 77736451, 93075032, 98151985, 65361399, 30424658, 40203420, 92532683, 24604526, 37012131, 82822261, 49453849, 28518142, 11228034, 26953184, 72651678, 47858928, 18980165, 10757863, 36013092, 71571049, 86988123, 38621510, 40217956, 40316012, 55442551, 72908533, 85899563, 55650191, 92954649, 23102420, 88403517, 12990036, 38505943, 75009164, 86020868, 7294934, 52901458, 26864312, 58084171, 61454302, 13745823, 49265198, 35117338, 96542327, 3511992, 54452180, 24703084, 98981554, 4632025, 57393295, 21794152, 79532948, 83347308, 42189307, 24175681, 81635264, 20169860, 91830823, 54044341, 47316787, 14082493, 89194041, 80289240, 86114499, 21586627, 64253399, 56263761, 28314015, 31586944, 47083175, 10290334, 68391175, 26864780, 49204492, 30601829]
config['val_seeds'] = [74845232, 49284185, 42479234, 2450347, 89798532, 57622223, 82698192, 3515098, 25097611, 67245265, 8967223, 23675136, 19809452, 14296000, 14595930, 92508128, 79317391, 26330449, 14873887, 67571396, 28571056, 22706725, 6867935, 30775713, 65648694, 77834471, 70727827, 54730261, 57391127, 91606755, 7073097, 34435218, 99777057, 13287272, 82423362, 13696706, 89790540, 59003772, 12112245, 57412363, 5519091, 64027712, 65587799, 75895447, 17647353, 40782712, 91367779, 64930082, 67592059, 77193383, 74885844, 13674427, 99528266, 65226388, 61261202, 55898502, 98991159, 99958647, 50435368, 10202150, 66673666, 68367198, 63034145, 65755534, 9359950, 72468035, 62898684, 21266089, 33564123, 83820918, 62318525, 96309953, 92128025, 53682023, 60434213, 72192258, 13110668, 6346147, 49893049, 65779991, 31344345, 10368829, 17581615, 41676915, 9793031, 93525062, 70079396, 78295191, 90575207, 28597308, 6380637, 25152574, 9559890, 95091874, 76648481, 33900209, 54357812, 41889952, 63060188, 56012838]
config['test_seed'] = 60740054

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/city_scapes_c/pspnet_resnet50/single_task_depth_estimation_run_1"
