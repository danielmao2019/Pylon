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
    'work_dir': './logs/benchmarks/point_cloud_registration/synth_pcr_0.5/BUFFER_run_1',
    'epochs': 100,
    'init_seed': 32611073,
    'train_seeds': [61616609, 49341628, 98011890, 38053985, 189863, 9142055, 97158482, 61723023, 91413598, 28940822, 46406028, 39841057, 75500318, 72375211, 5938914, 9030727, 95774988, 95635915, 87791125, 81192810, 96835151, 12843904, 90629446, 24028092, 43180687, 94035220, 66598449, 41452650, 59688438, 65856741, 46056302, 35650707, 28819290, 3289557, 12648464, 77289331, 78286045, 56469969, 900596, 29738662, 39070606, 48055661, 76166465, 84222508, 10542968, 10630044, 91666050, 78412921, 38297742, 92989983, 209940, 65865556, 9352016, 91091233, 23487379, 3464376, 47774816, 55814219, 35886129, 69577970, 56093226, 68954034, 22356855, 23761258, 76447665, 376502, 42612205, 38554598, 65426340, 73033663, 78388511, 89886888, 93853238, 15026872, 83133400, 59163641, 64641973, 9903663, 94024809, 49057749, 20689690, 33716318, 30096124, 98562561, 49553489, 90546816, 72552042, 34612442, 62836618, 17006575, 91170582, 70527933, 49772814, 75025150, 31956092, 42790075, 39959825, 44680627, 44937869, 87675815],
    'val_seeds': [61114836, 73323044, 90480116, 15254849, 73184952, 38937362, 20121317, 14514771, 96224185, 99926002, 94793281, 19045149, 20687432, 31380070, 66604291, 73870516, 74916136, 82556131, 9423205, 32216736, 63748573, 86325072, 71182949, 58509560, 29907216, 27465227, 6613150, 39945989, 72165799, 94479725, 16209109, 7332588, 77085319, 51891919, 63086294, 88075966, 4084809, 74271985, 24369128, 43146817, 93936717, 48047783, 67570649, 72126584, 97161734, 55039116, 3559280, 13229245, 31040726, 65587866, 74587916, 98482497, 97133759, 77133622, 51886462, 38952760, 82084348, 36178878, 18964296, 37037830, 15271726, 91945813, 9883644, 96070336, 43487627, 20697464, 5541379, 44802836, 69379852, 63699197, 1780346, 33433131, 49951482, 93605003, 70086530, 13860710, 42580284, 81685657, 57972726, 82881699, 23726730, 6456990, 7381965, 492241, 27045373, 73900746, 36999901, 15657398, 30012145, 28251909, 41080365, 70851056, 76223340, 40196154, 58629073, 51608852, 17105529, 31496175, 24158949, 2941986],
    'test_seed': 63551089,
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
    'work_dir': './logs/benchmarks/point_cloud_registration/synth_pcr_0.5/BUFFER_run_1',
    'epochs': 100,
    'init_seed': 48103225,
    'train_seeds': [22812761, 64316320, 48511758, 75692465, 12557115, 80929953, 43915947, 79617829, 66118217, 32492003, 89938625, 74493734, 52175849, 78138265, 20642294, 81235683, 79358633, 3201813, 11893136, 51776554, 93703942, 18460656, 19351922, 41223004, 7472911, 20554645, 94730591, 31186272, 88522387, 74672959, 52374096, 82392361, 48750998, 59622399, 13399767, 98033258, 20908165, 4880435, 77447603, 53477456, 5150970, 63913934, 31920324, 53890563, 62416022, 73149914, 99680835, 98789700, 7928729, 65893024, 36878982, 94412866, 31953435, 13775808, 81915237, 46309929, 12185346, 44284642, 20254446, 50645594, 77722451, 31325218, 60168905, 80625297, 23124267, 88509999, 80892067, 43027125, 53999371, 38029881, 48495188, 92885207, 87653863, 75780391, 37419438, 74082302, 54849906, 60024474, 48988219, 9948762, 63489227, 67986324, 77770052, 85115153, 80831776, 67864851, 4603699, 34514268, 45767710, 83117112, 9881973, 26570700, 62538544, 77301305, 9443958, 93675903, 81217175, 68287389, 72290913, 67922706],
    'val_seeds': [36075159, 84776359, 56021749, 9555700, 94997064, 68302340, 97676722, 34953399, 90123158, 34050264, 77238737, 67802928, 50786252, 14649553, 82931089, 60417850, 92441156, 44047737, 65226653, 52418504, 67001364, 7751401, 87204846, 37550712, 90357695, 19769665, 82624678, 45296724, 3541425, 91917027, 59967309, 61251160, 73131173, 10612045, 5643899, 77542466, 1725125, 34642449, 79574094, 95431443, 75965005, 91865495, 47180135, 23014194, 94762585, 5988849, 20314565, 86910178, 94330796, 45941476, 72943284, 10400801, 75776724, 79540148, 93234991, 95185561, 85408373, 36384772, 49238859, 95145987, 53971273, 21703063, 65863201, 29682277, 68560838, 74869145, 98989668, 65785694, 21163574, 35560588, 26030981, 86643426, 16939974, 52930197, 94892678, 26878437, 69834586, 95935934, 5408687, 93920883, 20459271, 91527793, 10784264, 84724062, 4965988, 73205810, 36479596, 71972516, 79695726, 22475858, 5763140, 11643908, 8391814, 61909695, 28754802, 23962108, 85706935, 17953147, 37513642, 51876937],
    'test_seed': 77444235,
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
    'work_dir': './logs/benchmarks/point_cloud_registration/synth_pcr_0.5/BUFFER_run_1',
    'epochs': 100,
    'init_seed': 63555337,
    'train_seeds': [21185687, 9769369, 80337235, 48734720, 12551504, 59789739, 65320058, 82899347, 40876702, 58132829, 34923008, 39039812, 18658372, 59661759, 12964735, 29319302, 44943863, 88256049, 17948206, 8251825, 3599562, 68057826, 63451575, 65849540, 72791220, 52245335, 51429769, 86590195, 26086506, 98772328, 18134727, 36937920, 56320093, 14478949, 91483027, 67778975, 68908666, 85179264, 97781359, 54326829, 8257073, 45729681, 53398706, 8055317, 73928559, 75963646, 47895478, 59278908, 26561369, 67800560, 89828309, 50830727, 96326293, 909679, 26592700, 66361171, 18714960, 38923502, 66180527, 94653595, 66979597, 11747308, 64558166, 80936533, 80639638, 52837658, 13604093, 33351646, 84884623, 92271000, 11951396, 25688556, 99854112, 40182771, 81250048, 79515504, 65385318, 69735332, 14740518, 41639462, 88614068, 66079258, 33627309, 5800525, 10730105, 82139968, 26126183, 77556017, 32289860, 49043405, 51746150, 13043839, 31089314, 51624747, 81941290, 51414097, 17623753, 72464559, 90938710, 8886316],
    'val_seeds': [53343749, 46650498, 86204173, 20828485, 32574376, 95841181, 79116219, 18493146, 33292967, 7344379, 78069205, 68948278, 48525115, 88661419, 92120467, 31580000, 1278042, 29742202, 20436193, 8422301, 78981563, 4951464, 24906023, 17179211, 55429778, 49364688, 48305237, 20543559, 55062808, 35276845, 51772243, 36818087, 73216314, 91328908, 84428936, 70745473, 61223735, 41359717, 29194457, 81399756, 25655555, 24129453, 4655583, 56022547, 60373714, 26556008, 91656663, 70680931, 44798840, 53945492, 70015264, 42428179, 87805892, 1752247, 4783557, 58193550, 89468440, 57138079, 4704714, 15705764, 41047826, 96050006, 74169566, 25248510, 63251072, 47370932, 85877381, 45069322, 7850922, 22860370, 44532771, 78329165, 79638712, 62494650, 6708794, 61140848, 66498114, 9802214, 25948367, 37111597, 13290459, 26215206, 42853056, 51823071, 31751388, 68571778, 54295451, 41017986, 58719685, 20051725, 40267220, 72521741, 30735684, 20353087, 94609833, 62788863, 31031211, 42930237, 52255738, 14987330],
    'test_seed': 72319593,
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
    'work_dir': './logs/benchmarks/point_cloud_registration/synth_pcr_0.5/BUFFER_run_1',
    'epochs': 100,
    'init_seed': 22931043,
    'train_seeds': [59987140, 84520516, 89512391, 76868279, 96957643, 49783595, 67752243, 28335436, 5302539, 19389368, 41260137, 35730212, 12158547, 57879708, 85709087, 15473844, 53212955, 99612008, 20927122, 74221476, 44778613, 37807337, 90428995, 53104409, 95648642, 44818131, 49451548, 84762082, 41821270, 50654388, 48587246, 1512067, 7111850, 9734572, 32920217, 85324871, 94959428, 49480924, 95384476, 91480466, 1722998, 42550258, 73535228, 45150509, 41619182, 20502265, 21682778, 52819267, 32867689, 43722937, 679460, 74171774, 51708825, 25410847, 45863651, 71286416, 36076743, 82889877, 35140869, 18683703, 42764543, 31282299, 75697653, 93453562, 68307674, 61293666, 31714313, 74861089, 50815928, 52611168, 94559298, 79554054, 45338502, 25477074, 19825485, 64740186, 2891978, 49829752, 84255956, 36573712, 39940359, 65062910, 84370722, 41374675, 91931247, 45003528, 46293065, 52251303, 75980645, 86997034, 12099011, 30444899, 43910059, 58027889, 97184172, 53340876, 98748707, 34700114, 71186982, 48610339],
    'val_seeds': [97721448, 32988116, 53041333, 49638532, 82858691, 32999788, 32445117, 85672621, 60497418, 21992070, 32859159, 84308791, 59767937, 42133969, 96656572, 1835045, 2462553, 86719772, 56727565, 52421183, 41112562, 5578528, 1475051, 46935702, 16844933, 82038062, 95169487, 39500426, 17360491, 18640737, 4613923, 45251966, 87976110, 20702802, 15301086, 97460519, 1557365, 36810318, 9516519, 48881452, 41830973, 97855392, 25189936, 68824311, 76528982, 83049483, 23119750, 93361122, 39814264, 69275482, 18201239, 40123256, 70520396, 49356307, 15370316, 13452095, 75708708, 98572287, 36613779, 95114191, 82095351, 61047636, 27460080, 37562277, 4815801, 72608270, 77743965, 70058852, 81111084, 72781803, 8174244, 1710080, 27700557, 6757536, 55269933, 47132849, 36581554, 935048, 73755645, 82857056, 15840457, 39833617, 22954920, 42937619, 74405655, 83881008, 21880440, 67992945, 54895691, 24075886, 36418814, 94100212, 12932300, 96350217, 60780764, 17771441, 980225, 70163896, 94360901, 13029627],
    'test_seed': 52891401,
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
