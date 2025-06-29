# This file is automatically generated by `./configs/benchmarks/point_cloud_registration/gen.py`.
# Please do not attempt to modify manually.
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR
from criteria.vision_3d.point_cloud_registration.buffer_criteria.desc_stage_criterion import BUFFER_DescStageCriterion
from criteria.vision_3d.point_cloud_registration.buffer_criteria.inlier_stage_criterion import BUFFER_InlierStageCriterion
from criteria.vision_3d.point_cloud_registration.buffer_criteria.keypt_stage_criterion import BUFFER_KeyptStageCriterion
from criteria.vision_3d.point_cloud_registration.buffer_criteria.ref_stage_criterion import BUFFER_RefStageCriterion
from data.dataloaders.buffer_dataloader import BufferDataloader
from data.datasets.pcr_datasets.kitti_dataset import KITTIDataset
from data.transforms.compose import Compose
from data.transforms.identity import Identity
from data.transforms.vision_3d.clamp import Clamp
from data.transforms.vision_3d.downsample import DownSample
from data.transforms.vision_3d.estimate_normals import EstimateNormals
from data.transforms.vision_3d.random_rigid_transform import RandomRigidTransform
from data.transforms.vision_3d.shuffle import Shuffle
from data.transforms.vision_3d.uniform_pos_noise import UniformPosNoise
from metrics.vision_3d.point_cloud_registration.buffer_metrics.desc_stage_metric import BUFFER_DescStageMetric
from metrics.vision_3d.point_cloud_registration.buffer_metrics.inlier_stage_metric import BUFFER_InlierStageMetric
from metrics.vision_3d.point_cloud_registration.buffer_metrics.keypt_stage_metric import BUFFER_KeyptStageMetric
from metrics.vision_3d.point_cloud_registration.buffer_metrics.ref_stage_metric import BUFFER_RefStageMetric
from models.point_cloud_registration.buffer.buffer import BUFFER
from optimizers.single_task_optimizer import SingleTaskOptimizer
from runners.pcr_trainers.buffer_trainer import BufferTrainer


config = [{
    'stage': 'Ref',
    'runner': BufferTrainer,
    'work_dir': './logs/benchmarks/point_cloud_registration/real_pcr_1.0/BUFFER_run_1',
    'epochs': 100,
    'init_seed': 22742994,
    'train_seeds': [74794441, 68782431, 23570604, 39887516, 46006206, 8852659, 8025695, 97063607, 14981770, 70835780, 65162896, 33268043, 22684056, 27576367, 14844015, 17541586, 81464271, 24795467, 22140193, 84136259, 65293040, 40123182, 87619287, 87719798, 57530541, 29274784, 23969849, 93255535, 48615633, 41451978, 69212117, 62419168, 84006088, 74481868, 62123342, 52449273, 77653803, 40692883, 96444887, 82416968, 4956734, 87340463, 98869627, 87850587, 8705678, 94757890, 89134296, 15478772, 50929998, 83014079, 179730, 41800493, 3528526, 51788884, 55897772, 90686485, 10170570, 54629211, 32222986, 65811501, 67793262, 83374671, 52484664, 87114354, 99030180, 86197579, 7798640, 84157069, 98525333, 33888342, 66012858, 39739569, 92579626, 79279770, 49078489, 83474783, 4414484, 82638330, 55907895, 4193832, 46492327, 24547884, 82601578, 1478984, 94107737, 32450954, 33933212, 95523974, 59783304, 95590414, 69261851, 99640796, 32346998, 94201400, 65538084, 97537308, 23984719, 65219824, 86805246, 11941737],
    'val_seeds': [43976498, 67492888, 21872635, 14672434, 89337670, 19215154, 85687239, 34343891, 74665265, 76200921, 51416538, 75778620, 60339958, 34039036, 11144960, 5112268, 52451573, 1060612, 87233230, 97447936, 1454890, 56808369, 68041383, 38077920, 25695782, 12129387, 55350378, 37858446, 28108721, 55688198, 56698072, 44653907, 17544234, 10191578, 36664349, 84530451, 83510479, 59439506, 92109138, 71271811, 88317418, 46117179, 50219214, 59088742, 96157264, 15453335, 88885167, 23986279, 29356196, 11365965, 33271429, 6794220, 34453460, 83368540, 68238876, 49320764, 51674039, 83686860, 19641591, 55567881, 3526102, 81890130, 9294334, 46249035, 67332919, 77434976, 43135495, 52473344, 29453565, 56271660, 57400316, 25070690, 84390906, 6216707, 30173965, 30406471, 57630263, 20441538, 83015098, 52236536, 76651213, 42865780, 28685839, 10788415, 76119663, 51704373, 69901613, 49155369, 46018913, 56327482, 75030740, 88475858, 12607012, 25581257, 88092982, 25581411, 54256064, 26678002, 84555168, 43989527],
    'test_seed': 49234590,
    'train_dataset': {
        'class': KITTIDataset,
        'args': {
            'data_root': './data/datasets/soft_links/KITTI',
            'split': 'train',
            'transforms_cfg': {
                'class': Compose,
                'args': {
                    'transforms': [{
    'op': {
        'class': UniformPosNoise,
        'args': {
            'min': -0.025,
            'max': 0.025,
        },
    },
    'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
}, {
    'op': {
        'class': DownSample,
        'args': {
            'voxel_size': 0.05,
        },
    },
    'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
    'output_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds')],
}, {
    'op': {
        'class': Shuffle,
        'args': {},
    },
    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds')],
}, {
    'op': {
        'class': RandomRigidTransform,
        'args': {
            'rot_mag': 180.0,
            'trans_mag': 0.0,
            'method': 'Euler',
            'num_axis': 3,
        },
    },
    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds'), ('labels', 'transform')],
}, {
    'op': {
        'class': DownSample,
        'args': {
            'voxel_size': 0.3,
        },
    },
    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds')],
    'output_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
}, {
    'op': {
        'class': Shuffle,
        'args': {},
    },
    'input_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
}, {
    'op': {
        'class': Clamp,
        'args': {
            'max_points': 40000,
        },
    },
    'input_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
}, {
    'op': {
        'class': EstimateNormals,
        'args': {},
    },
    'input_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
}, {
    'op': {
        'class': Identity,
        'args': {},
    },
    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds'), ('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds'), ('labels', 'transform'), ('meta_info', 'seq'), ('meta_info', 't0'), ('meta_info', 't1')],
}],
                },
            },
        },
    },
    'train_dataloader': {
        'class': BufferDataloader,
        'args': {
            'config': {
                'point': {
                    'conv_radius': 2.0,
                },
                'data': {
                    'voxel_size_0': 0.3,
                },
            },
            'batch_size': 1,
        },
    },
    'criterion': {
        'class': BUFFER_RefStageCriterion,
        'args': {},
    },
    'val_dataset': {
        'class': KITTIDataset,
        'args': {
            'data_root': './data/datasets/soft_links/KITTI',
            'split': 'val',
            'transforms_cfg': {
                'class': Compose,
                'args': {
                    'transforms': [{
    'op': {
        'class': UniformPosNoise,
        'args': {
            'min': -0.025,
            'max': 0.025,
        },
    },
    'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
}, {
    'op': {
        'class': DownSample,
        'args': {
            'voxel_size': 0.05,
        },
    },
    'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
    'output_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds')],
}, {
    'op': {
        'class': Shuffle,
        'args': {},
    },
    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds')],
}, {
    'op': {
        'class': RandomRigidTransform,
        'args': {
            'rot_mag': 180.0,
            'trans_mag': 0.0,
            'method': 'Euler',
            'num_axis': 3,
        },
    },
    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds'), ('labels', 'transform')],
}, {
    'op': {
        'class': DownSample,
        'args': {
            'voxel_size': 0.3,
        },
    },
    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds')],
    'output_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
}, {
    'op': {
        'class': Shuffle,
        'args': {},
    },
    'input_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
}, {
    'op': {
        'class': Clamp,
        'args': {
            'max_points': 40000,
        },
    },
    'input_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
}, {
    'op': {
        'class': EstimateNormals,
        'args': {},
    },
    'input_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
}, {
    'op': {
        'class': Identity,
        'args': {},
    },
    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds'), ('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds'), ('labels', 'transform'), ('meta_info', 'seq'), ('meta_info', 't0'), ('meta_info', 't1')],
}],
                },
            },
        },
    },
    'val_dataloader': {
        'class': BufferDataloader,
        'args': {
            'config': {
                'point': {
                    'conv_radius': 2.0,
                },
                'data': {
                    'voxel_size_0': 0.3,
                },
            },
            'batch_size': 1,
        },
    },
    'test_dataset': None,
    'test_dataloader': None,
    'metric': {
        'class': BUFFER_RefStageMetric,
        'args': {},
    },
    'model': {
        'class': BUFFER,
        'args': {
            'config': {
                'stage': 'Ref',
                'data': {
                    'voxel_size_0': 0.3,
                    'dataset': 'KITTI',
                },
                'train': {
                    'pos_num': 512,
                },
                'test': {
                    'scale': 1.0,
                    'pose_refine': False,
                },
                'point': {
                    'in_feats_dim': 3,
                    'first_feats_dim': 32,
                    'conv_radius': 2.0,
                    'keypts_th': 0.5,
                    'num_keypts': 1500,
                },
                'patch': {
                    'des_r': 3.0,
                    'num_points_per_patch': 512,
                    'rad_n': 3,
                    'azi_n': 20,
                    'ele_n': 7,
                    'delta': 0.8,
                    'voxel_sample': 10,
                },
                'match': {
                    'dist_th': 0.3,
                    'inlier_th': 2.0,
                    'similar_th': 0.9,
                    'confidence': 1.0,
                    'iter_n': 50000,
                },
            },
        },
    },
    'optimizer': {
        'class': SingleTaskOptimizer,
        'args': {
            'optimizer_config': {
                'class': Adam,
                'args': {
                    'params': None,
                    'lr': 0.0001,
                    'weight_decay': 1e-06,
                },
            },
        },
    },
    'scheduler': {
        'class': StepLR,
        'args': {
            'optimizer': None,
            'step_size': 1000,
            'gamma': 0.95,
        },
    },
}, {
    'stage': 'Desc',
    'runner': BufferTrainer,
    'work_dir': './logs/benchmarks/point_cloud_registration/real_pcr_1.0/BUFFER_run_1',
    'epochs': 100,
    'init_seed': 97462296,
    'train_seeds': [40122649, 1590880, 98464559, 55044916, 61985342, 48776396, 57438057, 18434016, 38425961, 99308891, 84128055, 67254831, 6199136, 40472093, 31622686, 98007824, 24737817, 6239186, 73181494, 91574890, 1653784, 45688251, 71347255, 76077012, 25427557, 6857343, 62758445, 50705871, 42975553, 40922908, 77297805, 58213064, 89742531, 62609280, 29953245, 82833573, 41043876, 28413413, 12285162, 38936836, 46978570, 65410807, 19963935, 80213525, 70499030, 88379756, 87856662, 97815893, 72366800, 56491909, 6769945, 53299349, 53973111, 40355996, 63598767, 55935406, 92270198, 37416193, 4121320, 77950996, 32553492, 91752939, 10801124, 51096319, 42961795, 51200326, 33737804, 27375871, 86027797, 49143899, 52430069, 98366234, 94412646, 60277951, 40160947, 47894464, 48727314, 11780882, 87234130, 21521124, 66718407, 39477271, 66977276, 76147624, 87246791, 75787344, 88046481, 27587674, 99250230, 46862162, 89040560, 70000105, 35561430, 62612743, 22004746, 68612697, 24705505, 35913560, 22056455, 54080741],
    'val_seeds': [65367885, 5646775, 93882175, 32900775, 31815694, 90790499, 52501090, 29073169, 20402502, 32696729, 21301158, 74707174, 46764767, 58219642, 5776880, 1807222, 24750165, 2680384, 35085113, 45308918, 96400654, 4967272, 12261889, 69248731, 4036019, 2758490, 6738805, 59165910, 40120268, 75727736, 93517071, 89524089, 57347562, 87200837, 60297727, 99390275, 2437981, 51182275, 87303513, 23160288, 16824087, 89279586, 18108551, 11869428, 76461816, 2083531, 49420700, 86517779, 94667980, 58671304, 69002203, 48473121, 72367723, 36827233, 92054025, 76042090, 1838831, 32596716, 14430649, 97739368, 83479018, 92940461, 99282946, 89461383, 73703460, 50955777, 77452554, 60873446, 54388656, 12316264, 94578114, 4067926, 23616381, 16711062, 49720972, 22471573, 42804102, 82746749, 13967765, 12957931, 38421040, 10107334, 3566556, 10209107, 16296728, 47771091, 12549740, 72734998, 39369398, 35252605, 94327151, 5439333, 6208297, 64617677, 80998144, 20740970, 81880276, 42824938, 25414628, 18775544],
    'test_seed': 86799898,
    'train_dataset': {
        'class': KITTIDataset,
        'args': {
            'data_root': './data/datasets/soft_links/KITTI',
            'split': 'train',
            'transforms_cfg': {
                'class': Compose,
                'args': {
                    'transforms': [{
    'op': {
        'class': UniformPosNoise,
        'args': {
            'min': -0.025,
            'max': 0.025,
        },
    },
    'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
}, {
    'op': {
        'class': DownSample,
        'args': {
            'voxel_size': 0.05,
        },
    },
    'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
    'output_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds')],
}, {
    'op': {
        'class': Shuffle,
        'args': {},
    },
    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds')],
}, {
    'op': {
        'class': RandomRigidTransform,
        'args': {
            'rot_mag': 180.0,
            'trans_mag': 0.0,
            'method': 'Euler',
            'num_axis': 1,
        },
    },
    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds'), ('labels', 'transform')],
}, {
    'op': {
        'class': DownSample,
        'args': {
            'voxel_size': 0.3,
        },
    },
    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds')],
    'output_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
}, {
    'op': {
        'class': Shuffle,
        'args': {},
    },
    'input_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
}, {
    'op': {
        'class': Clamp,
        'args': {
            'max_points': 40000,
        },
    },
    'input_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
}, {
    'op': {
        'class': EstimateNormals,
        'args': {},
    },
    'input_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
}, {
    'op': {
        'class': Identity,
        'args': {},
    },
    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds'), ('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds'), ('labels', 'transform'), ('meta_info', 'seq'), ('meta_info', 't0'), ('meta_info', 't1')],
}],
                },
            },
        },
    },
    'train_dataloader': {
        'class': BufferDataloader,
        'args': {
            'config': {
                'point': {
                    'conv_radius': 2.0,
                },
                'data': {
                    'voxel_size_0': 0.3,
                },
            },
            'batch_size': 1,
        },
    },
    'criterion': {
        'class': BUFFER_DescStageCriterion,
        'args': {},
    },
    'val_dataset': {
        'class': KITTIDataset,
        'args': {
            'data_root': './data/datasets/soft_links/KITTI',
            'split': 'val',
            'transforms_cfg': {
                'class': Compose,
                'args': {
                    'transforms': [{
    'op': {
        'class': UniformPosNoise,
        'args': {
            'min': -0.025,
            'max': 0.025,
        },
    },
    'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
}, {
    'op': {
        'class': DownSample,
        'args': {
            'voxel_size': 0.05,
        },
    },
    'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
    'output_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds')],
}, {
    'op': {
        'class': Shuffle,
        'args': {},
    },
    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds')],
}, {
    'op': {
        'class': RandomRigidTransform,
        'args': {
            'rot_mag': 180.0,
            'trans_mag': 0.0,
            'method': 'Euler',
            'num_axis': 1,
        },
    },
    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds'), ('labels', 'transform')],
}, {
    'op': {
        'class': DownSample,
        'args': {
            'voxel_size': 0.3,
        },
    },
    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds')],
    'output_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
}, {
    'op': {
        'class': Shuffle,
        'args': {},
    },
    'input_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
}, {
    'op': {
        'class': Clamp,
        'args': {
            'max_points': 40000,
        },
    },
    'input_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
}, {
    'op': {
        'class': EstimateNormals,
        'args': {},
    },
    'input_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
}, {
    'op': {
        'class': Identity,
        'args': {},
    },
    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds'), ('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds'), ('labels', 'transform'), ('meta_info', 'seq'), ('meta_info', 't0'), ('meta_info', 't1')],
}],
                },
            },
        },
    },
    'val_dataloader': {
        'class': BufferDataloader,
        'args': {
            'config': {
                'point': {
                    'conv_radius': 2.0,
                },
                'data': {
                    'voxel_size_0': 0.3,
                },
            },
            'batch_size': 1,
        },
    },
    'test_dataset': None,
    'test_dataloader': None,
    'metric': {
        'class': BUFFER_DescStageMetric,
        'args': {},
    },
    'model': {
        'class': BUFFER,
        'args': {
            'config': {
                'stage': 'Desc',
                'data': {
                    'voxel_size_0': 0.3,
                    'dataset': 'KITTI',
                },
                'train': {
                    'pos_num': 512,
                },
                'test': {
                    'scale': 1.0,
                    'pose_refine': False,
                },
                'point': {
                    'in_feats_dim': 3,
                    'first_feats_dim': 32,
                    'conv_radius': 2.0,
                    'keypts_th': 0.5,
                    'num_keypts': 1500,
                },
                'patch': {
                    'des_r': 3.0,
                    'num_points_per_patch': 512,
                    'rad_n': 3,
                    'azi_n': 20,
                    'ele_n': 7,
                    'delta': 0.8,
                    'voxel_sample': 10,
                },
                'match': {
                    'dist_th': 0.3,
                    'inlier_th': 2.0,
                    'similar_th': 0.9,
                    'confidence': 1.0,
                    'iter_n': 50000,
                },
            },
        },
    },
    'optimizer': {
        'class': SingleTaskOptimizer,
        'args': {
            'optimizer_config': {
                'class': Adam,
                'args': {
                    'params': None,
                    'lr': 0.0001,
                    'weight_decay': 1e-06,
                },
            },
        },
    },
    'scheduler': {
        'class': StepLR,
        'args': {
            'optimizer': None,
            'step_size': 1000,
            'gamma': 0.95,
        },
    },
}, {
    'stage': 'Keypt',
    'runner': BufferTrainer,
    'work_dir': './logs/benchmarks/point_cloud_registration/real_pcr_1.0/BUFFER_run_1',
    'epochs': 100,
    'init_seed': 80567727,
    'train_seeds': [40966865, 82787909, 58535887, 64012163, 5735711, 98628695, 71903288, 653886, 10395352, 53365369, 84853135, 92647569, 43691874, 60608325, 41421907, 9381820, 95868149, 20890036, 31859401, 78951047, 61394142, 55844205, 99441819, 58184529, 47817789, 24345677, 43203452, 17365739, 56991778, 26774322, 63717797, 8574704, 21358534, 1895656, 76161074, 57549483, 12535682, 17823188, 7312951, 53094622, 11507596, 98604637, 38422202, 28125836, 66129086, 1391772, 97311813, 45864429, 83794635, 16776530, 95256073, 47092211, 27844046, 37861834, 8092159, 58361511, 96375262, 91721221, 93665441, 32696029, 17885246, 89220122, 58392627, 46105962, 18188653, 5911167, 85945569, 75912992, 67955900, 26613903, 77100724, 55557907, 85367630, 9887553, 91724772, 88165144, 66471592, 35914820, 81515615, 45818927, 45327617, 46655640, 44007324, 26382090, 66652662, 79987424, 8344232, 6545780, 46101567, 65019578, 68052106, 65175747, 37924941, 40048813, 23393224, 43403766, 28051551, 40958485, 76127645, 7463138],
    'val_seeds': [50991073, 11479546, 15772680, 88295885, 47611579, 77821523, 77222521, 30077041, 31031350, 70575437, 85910353, 48810599, 53580642, 81142828, 22839935, 92155370, 40909617, 8755694, 79061787, 97261469, 39038859, 8919998, 6228801, 61416474, 1546639, 45748954, 5458876, 15399575, 68184553, 820519, 28641450, 50790883, 29243706, 80772406, 37135509, 16926899, 22941192, 49660350, 77366407, 14128899, 58464296, 99094423, 73571029, 83480902, 26484730, 72925558, 90417694, 99188021, 88656488, 15740457, 82876085, 74033042, 85818572, 40597313, 86430663, 25400298, 8159131, 21885788, 19850060, 7113403, 47684352, 68118970, 91651726, 70484482, 23563376, 22416389, 56399893, 31338979, 54696796, 24779032, 47887732, 60072229, 76870674, 51559802, 27479673, 12566956, 52051045, 18434381, 2998051, 34936877, 80635889, 49782046, 42240091, 39424383, 67301839, 47504692, 78645073, 3669699, 24011268, 47839165, 4478267, 57146928, 31954521, 6963122, 84202448, 8678428, 58690439, 94341283, 20810380, 47830401],
    'test_seed': 59184533,
    'train_dataset': {
        'class': KITTIDataset,
        'args': {
            'data_root': './data/datasets/soft_links/KITTI',
            'split': 'train',
            'transforms_cfg': {
                'class': Compose,
                'args': {
                    'transforms': [{
    'op': {
        'class': UniformPosNoise,
        'args': {
            'min': -0.025,
            'max': 0.025,
        },
    },
    'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
}, {
    'op': {
        'class': DownSample,
        'args': {
            'voxel_size': 0.05,
        },
    },
    'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
    'output_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds')],
}, {
    'op': {
        'class': Shuffle,
        'args': {},
    },
    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds')],
}, {
    'op': {
        'class': RandomRigidTransform,
        'args': {
            'rot_mag': 180.0,
            'trans_mag': 0.0,
            'method': 'Euler',
            'num_axis': 1,
        },
    },
    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds'), ('labels', 'transform')],
}, {
    'op': {
        'class': DownSample,
        'args': {
            'voxel_size': 0.3,
        },
    },
    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds')],
    'output_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
}, {
    'op': {
        'class': Shuffle,
        'args': {},
    },
    'input_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
}, {
    'op': {
        'class': Clamp,
        'args': {
            'max_points': 40000,
        },
    },
    'input_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
}, {
    'op': {
        'class': EstimateNormals,
        'args': {},
    },
    'input_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
}, {
    'op': {
        'class': Identity,
        'args': {},
    },
    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds'), ('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds'), ('labels', 'transform'), ('meta_info', 'seq'), ('meta_info', 't0'), ('meta_info', 't1')],
}],
                },
            },
        },
    },
    'train_dataloader': {
        'class': BufferDataloader,
        'args': {
            'config': {
                'point': {
                    'conv_radius': 2.0,
                },
                'data': {
                    'voxel_size_0': 0.3,
                },
            },
            'batch_size': 1,
        },
    },
    'criterion': {
        'class': BUFFER_KeyptStageCriterion,
        'args': {},
    },
    'val_dataset': {
        'class': KITTIDataset,
        'args': {
            'data_root': './data/datasets/soft_links/KITTI',
            'split': 'val',
            'transforms_cfg': {
                'class': Compose,
                'args': {
                    'transforms': [{
    'op': {
        'class': UniformPosNoise,
        'args': {
            'min': -0.025,
            'max': 0.025,
        },
    },
    'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
}, {
    'op': {
        'class': DownSample,
        'args': {
            'voxel_size': 0.05,
        },
    },
    'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
    'output_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds')],
}, {
    'op': {
        'class': Shuffle,
        'args': {},
    },
    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds')],
}, {
    'op': {
        'class': RandomRigidTransform,
        'args': {
            'rot_mag': 180.0,
            'trans_mag': 0.0,
            'method': 'Euler',
            'num_axis': 1,
        },
    },
    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds'), ('labels', 'transform')],
}, {
    'op': {
        'class': DownSample,
        'args': {
            'voxel_size': 0.3,
        },
    },
    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds')],
    'output_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
}, {
    'op': {
        'class': Shuffle,
        'args': {},
    },
    'input_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
}, {
    'op': {
        'class': Clamp,
        'args': {
            'max_points': 40000,
        },
    },
    'input_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
}, {
    'op': {
        'class': EstimateNormals,
        'args': {},
    },
    'input_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
}, {
    'op': {
        'class': Identity,
        'args': {},
    },
    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds'), ('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds'), ('labels', 'transform'), ('meta_info', 'seq'), ('meta_info', 't0'), ('meta_info', 't1')],
}],
                },
            },
        },
    },
    'val_dataloader': {
        'class': BufferDataloader,
        'args': {
            'config': {
                'point': {
                    'conv_radius': 2.0,
                },
                'data': {
                    'voxel_size_0': 0.3,
                },
            },
            'batch_size': 1,
        },
    },
    'test_dataset': None,
    'test_dataloader': None,
    'metric': {
        'class': BUFFER_KeyptStageMetric,
        'args': {},
    },
    'model': {
        'class': BUFFER,
        'args': {
            'config': {
                'stage': 'Keypt',
                'data': {
                    'voxel_size_0': 0.3,
                    'dataset': 'KITTI',
                },
                'train': {
                    'pos_num': 512,
                },
                'test': {
                    'scale': 1.0,
                    'pose_refine': False,
                },
                'point': {
                    'in_feats_dim': 3,
                    'first_feats_dim': 32,
                    'conv_radius': 2.0,
                    'keypts_th': 0.5,
                    'num_keypts': 1500,
                },
                'patch': {
                    'des_r': 3.0,
                    'num_points_per_patch': 512,
                    'rad_n': 3,
                    'azi_n': 20,
                    'ele_n': 7,
                    'delta': 0.8,
                    'voxel_sample': 10,
                },
                'match': {
                    'dist_th': 0.3,
                    'inlier_th': 2.0,
                    'similar_th': 0.9,
                    'confidence': 1.0,
                    'iter_n': 50000,
                },
            },
        },
    },
    'optimizer': {
        'class': SingleTaskOptimizer,
        'args': {
            'optimizer_config': {
                'class': Adam,
                'args': {
                    'params': None,
                    'lr': 0.0001,
                    'weight_decay': 1e-06,
                },
            },
        },
    },
    'scheduler': {
        'class': StepLR,
        'args': {
            'optimizer': None,
            'step_size': 1000,
            'gamma': 0.95,
        },
    },
}, {
    'stage': 'Inlier',
    'runner': BufferTrainer,
    'work_dir': './logs/benchmarks/point_cloud_registration/real_pcr_1.0/BUFFER_run_1',
    'epochs': 100,
    'init_seed': 63647628,
    'train_seeds': [77820730, 40202054, 56381060, 16321275, 11867116, 15697053, 47657675, 53705607, 81733693, 82846471, 84787071, 1075968, 58938924, 48015612, 44311755, 65949941, 6247260, 7427350, 32121761, 97090768, 2432553, 69664110, 14708697, 38205701, 72897237, 49405769, 205667, 9213015, 66987586, 20308083, 40181215, 57113289, 9614218, 9290480, 5875888, 18106310, 29700773, 35034622, 57610255, 46370428, 85611210, 13416630, 54355243, 72941139, 79645014, 69317072, 53540579, 9964443, 13104755, 16162184, 29577726, 1270296, 57009407, 38114528, 27315056, 35170440, 16522337, 73709357, 70137188, 32170053, 78821475, 71051633, 73048409, 23627440, 86364655, 58165671, 43327023, 46987561, 2860827, 54850929, 62117894, 28705077, 77199104, 68794737, 86435074, 49952472, 68106920, 12275587, 69952114, 57636294, 90328224, 18485737, 82293215, 17495756, 54887632, 10690310, 2384272, 54797189, 91784043, 17335175, 73394838, 24854389, 62716303, 68441116, 91945325, 20787810, 85664906, 27522702, 42955301, 24177554],
    'val_seeds': [59018295, 77286916, 15320048, 85122231, 24852405, 17195128, 92252550, 30195789, 68899031, 2266426, 2476167, 41413157, 13226080, 20878491, 86131295, 72489107, 84431784, 87868843, 78367570, 21333440, 84657716, 2107478, 51928927, 2578119, 37890751, 1588162, 32647065, 44526630, 35736651, 51277519, 15207883, 86069585, 64593329, 67050407, 72684524, 56942897, 49942504, 99139148, 18545736, 2047043, 35022600, 58037685, 15644875, 5472008, 7864455, 3728612, 7857035, 12884523, 39071485, 4929753, 9055392, 52100672, 32809026, 85840140, 36479466, 84599162, 4629543, 81359671, 21907915, 79519704, 484995, 94636569, 5735576, 41750377, 72495374, 35082001, 38455342, 56782158, 44580668, 98470162, 79731325, 82186222, 55753014, 20806311, 37283889, 11563239, 46087985, 2086073, 31745417, 68716447, 92658207, 17309299, 69517708, 56853555, 71429503, 31892261, 93989884, 27157369, 32970231, 77160926, 51846623, 98320905, 9608478, 91882808, 87022962, 11047708, 14242249, 44419178, 83565715, 44106718],
    'test_seed': 65812328,
    'train_dataset': {
        'class': KITTIDataset,
        'args': {
            'data_root': './data/datasets/soft_links/KITTI',
            'split': 'train',
            'transforms_cfg': {
                'class': Compose,
                'args': {
                    'transforms': [{
    'op': {
        'class': UniformPosNoise,
        'args': {
            'min': -0.025,
            'max': 0.025,
        },
    },
    'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
}, {
    'op': {
        'class': DownSample,
        'args': {
            'voxel_size': 0.05,
        },
    },
    'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
    'output_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds')],
}, {
    'op': {
        'class': Shuffle,
        'args': {},
    },
    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds')],
}, {
    'op': {
        'class': RandomRigidTransform,
        'args': {
            'rot_mag': 180.0,
            'trans_mag': 0.0,
            'method': 'Euler',
            'num_axis': 1,
        },
    },
    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds'), ('labels', 'transform')],
}, {
    'op': {
        'class': DownSample,
        'args': {
            'voxel_size': 0.3,
        },
    },
    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds')],
    'output_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
}, {
    'op': {
        'class': Shuffle,
        'args': {},
    },
    'input_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
}, {
    'op': {
        'class': Clamp,
        'args': {
            'max_points': 40000,
        },
    },
    'input_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
}, {
    'op': {
        'class': EstimateNormals,
        'args': {},
    },
    'input_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
}, {
    'op': {
        'class': Identity,
        'args': {},
    },
    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds'), ('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds'), ('labels', 'transform'), ('meta_info', 'seq'), ('meta_info', 't0'), ('meta_info', 't1')],
}],
                },
            },
        },
    },
    'train_dataloader': {
        'class': BufferDataloader,
        'args': {
            'config': {
                'point': {
                    'conv_radius': 2.0,
                },
                'data': {
                    'voxel_size_0': 0.3,
                },
            },
            'batch_size': 1,
        },
    },
    'criterion': {
        'class': BUFFER_InlierStageCriterion,
        'args': {},
    },
    'val_dataset': {
        'class': KITTIDataset,
        'args': {
            'data_root': './data/datasets/soft_links/KITTI',
            'split': 'val',
            'transforms_cfg': {
                'class': Compose,
                'args': {
                    'transforms': [{
    'op': {
        'class': UniformPosNoise,
        'args': {
            'min': -0.025,
            'max': 0.025,
        },
    },
    'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
}, {
    'op': {
        'class': DownSample,
        'args': {
            'voxel_size': 0.05,
        },
    },
    'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
    'output_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds')],
}, {
    'op': {
        'class': Shuffle,
        'args': {},
    },
    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds')],
}, {
    'op': {
        'class': RandomRigidTransform,
        'args': {
            'rot_mag': 180.0,
            'trans_mag': 0.0,
            'method': 'Euler',
            'num_axis': 1,
        },
    },
    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds'), ('labels', 'transform')],
}, {
    'op': {
        'class': DownSample,
        'args': {
            'voxel_size': 0.3,
        },
    },
    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds')],
    'output_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
}, {
    'op': {
        'class': Shuffle,
        'args': {},
    },
    'input_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
}, {
    'op': {
        'class': Clamp,
        'args': {
            'max_points': 40000,
        },
    },
    'input_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
}, {
    'op': {
        'class': EstimateNormals,
        'args': {},
    },
    'input_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
}, {
    'op': {
        'class': Identity,
        'args': {},
    },
    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds'), ('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds'), ('labels', 'transform'), ('meta_info', 'seq'), ('meta_info', 't0'), ('meta_info', 't1')],
}],
                },
            },
        },
    },
    'val_dataloader': {
        'class': BufferDataloader,
        'args': {
            'config': {
                'point': {
                    'conv_radius': 2.0,
                },
                'data': {
                    'voxel_size_0': 0.3,
                },
            },
            'batch_size': 1,
        },
    },
    'test_dataset': None,
    'test_dataloader': None,
    'metric': {
        'class': BUFFER_InlierStageMetric,
        'args': {},
    },
    'model': {
        'class': BUFFER,
        'args': {
            'config': {
                'stage': 'Inlier',
                'data': {
                    'voxel_size_0': 0.3,
                    'dataset': 'KITTI',
                },
                'train': {
                    'pos_num': 512,
                },
                'test': {
                    'scale': 1.0,
                    'pose_refine': False,
                },
                'point': {
                    'in_feats_dim': 3,
                    'first_feats_dim': 32,
                    'conv_radius': 2.0,
                    'keypts_th': 0.5,
                    'num_keypts': 1500,
                },
                'patch': {
                    'des_r': 3.0,
                    'num_points_per_patch': 512,
                    'rad_n': 3,
                    'azi_n': 20,
                    'ele_n': 7,
                    'delta': 0.8,
                    'voxel_sample': 10,
                },
                'match': {
                    'dist_th': 0.3,
                    'inlier_th': 2.0,
                    'similar_th': 0.9,
                    'confidence': 1.0,
                    'iter_n': 50000,
                },
            },
        },
    },
    'optimizer': {
        'class': SingleTaskOptimizer,
        'args': {
            'optimizer_config': {
                'class': Adam,
                'args': {
                    'params': None,
                    'lr': 0.0001,
                    'weight_decay': 1e-06,
                },
            },
        },
    },
    'scheduler': {
        'class': StepLR,
        'args': {
            'optimizer': None,
            'step_size': 1000,
            'gamma': 0.95,
        },
    },
}]
