# This file is automatically generated by `./configs/benchmarks/change_detection/gen_change_star_v1.py`.
# Please do not attempt to modify manually.
from builtins import list
from configs.common.models.change_detection.change_star import Resnet50WithFPN
from torch import Tensor
from torch.optim.lr_scheduler import PolynomialLR
from torch.optim.sgd import SGD
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.transforms import ColorJitter
from criteria.vision_2d.change_detection.change_star_criterion import ChangeStarCriterion
from data.collators.base_collator import BaseCollator
from data.collators.change_star_collator import ChangeStarCollator
from data.datasets.change_detection_datasets.bi_temporal.air_change_dataset import AirChangeDataset
from data.datasets.change_detection_datasets.bi_temporal.cdd_dataset import CDDDataset
from data.datasets.change_detection_datasets.bi_temporal.levir_cd_dataset import LevirCdDataset
from data.datasets.change_detection_datasets.bi_temporal.oscd_dataset import OSCDDataset
from data.datasets.change_detection_datasets.bi_temporal.sysu_cd_dataset import SYSU_CD_Dataset
from data.datasets.change_detection_datasets.bi_temporal.xview2_dataset import xView2Dataset
from data.transforms.compose import Compose
from data.transforms.randomize import Randomize
from data.transforms.vision_2d.crop.random_crop import RandomCrop
from data.transforms.vision_2d.flip import Flip
from data.transforms.vision_2d.random_rotation import RandomRotation
from metrics.vision_2d.change_detection.change_star_metric import ChangeStarMetric
from models.change_detection.change_star.change_star import ChangeStar
from optimizers.single_task_optimizer import SingleTaskOptimizer
from runners.multi_val_dataset_trainer import MultiValDatasetTrainer


config = {
    'runner': MultiValDatasetTrainer,
    'work_dir': './logs/benchmarks/change_detection/change_star_v1/xview2_run_1',
    'epochs': 100,
    'init_seed': 15499791,
    'train_seeds': [85419771, 92804333, 57218812, 56791173, 75558854, 58458014, 20434790, 18042436, 42620765, 49881468, 87038457, 56885836, 84525157, 88687508, 63687656, 56826858, 80965661, 17473208, 87013701, 68158891, 87450963, 74581833, 27186070, 44994568, 39307962, 93701260, 98958996, 76437353, 52064584, 82900922, 34913150, 1797698, 4701575, 34402154, 2289980, 91170493, 26081855, 33757954, 23549168, 72871564, 69474205, 20309073, 55473578, 7344897, 64171092, 73970809, 99231599, 18793003, 16200759, 95569533, 86097835, 43545811, 21484835, 30631343, 3295128, 29390132, 32085498, 44611125, 60078730, 97245351, 17264523, 65536206, 389265, 84336210, 84350871, 31263833, 87678876, 76998967, 71200948, 29842505, 27471225, 45060990, 4546492, 9973942, 56885801, 30060936, 65753987, 76981895, 40392612, 28682419, 93911633, 11299292, 89944219, 58997781, 51485522, 47529120, 55349932, 30102046, 66406339, 74374644, 766454, 5236604, 49191896, 24872425, 72273722, 42067276, 28386830, 16759023, 69295115, 72060810],
    'val_seeds': [31134526, 39607078, 67683073, 59078853, 87214569, 52634884, 66235919, 40938139, 43385355, 49728743, 31871370, 71346848, 62922810, 30343503, 48516701, 34699894, 7068213, 99360396, 3543651, 98508770, 37030216, 34477799, 3047359, 31293670, 20129879, 57061024, 70938290, 74492596, 98157832, 22685717, 64244259, 53229226, 61742502, 24743298, 7690713, 59610403, 25495686, 35452010, 92584120, 7454504, 93416614, 45133174, 51785184, 97567964, 7008979, 29158384, 6382962, 82090288, 62411412, 80973500, 81002096, 85522727, 87702368, 88654272, 48782329, 25985639, 43154249, 45645677, 55917108, 34036813, 27983722, 93322881, 17087321, 13912559, 12623732, 4053260, 76188361, 23338427, 68852441, 22420296, 10373221, 59879877, 27110397, 74144504, 37225170, 84599752, 32839988, 40060009, 26980845, 43569665, 58804283, 87058531, 51960492, 54737387, 79849385, 35943885, 84835769, 97388701, 94479686, 84093891, 10406441, 68606502, 74243532, 20597994, 88027797, 3611949, 55412238, 25800638, 96906430, 46487430],
    'test_seed': 39204752,
    'train_dataset': {
        'class': xView2Dataset,
        'args': {
            'data_root': './data/datasets/soft_links/xView2',
            'split': 'train',
            'transforms_cfg': {
                'class': Compose,
                'args': {
                    'transforms': [(
    {
            'class': RandomCrop,
            'args': {
                'size': (224, 224),
            },
        },
    [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'lbl_1'), ('labels', 'lbl_2')]
), (
    {
            'class': RandomRotation,
            'args': {
                'choices': [0, 90, 180, 270],
            },
        },
    [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'lbl_1'), ('labels', 'lbl_2')]
), (
    {
            'class': Randomize,
            'args': {
                'transform': {
                    'class': Flip,
                    'args': {
                        'axis': -1,
                    },
                },
                'p': 0.5,
            },
        },
    [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'lbl_1'), ('labels', 'lbl_2')]
), (
    {
            'class': Randomize,
            'args': {
                'transform': {
                    'class': Flip,
                    'args': {
                        'axis': -2,
                    },
                },
                'p': 0.5,
            },
        },
    [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'lbl_1'), ('labels', 'lbl_2')]
), (
    {
            'class': Randomize,
            'args': {
                'transform': {
                    'class': ColorJitter,
                    'args': {
                        'brightness': 0.5,
                        'contrast': 0.5,
                        'saturation': 0.5,
                    },
                },
                'p': 0.5,
            },
        },
    ('inputs', 'img_1')
), (
    {
            'class': Randomize,
            'args': {
                'transform': {
                    'class': ColorJitter,
                    'args': {
                        'brightness': 0.5,
                        'contrast': 0.5,
                        'saturation': 0.5,
                    },
                },
                'p': 0.5,
            },
        },
    ('inputs', 'img_2')
)],
                },
            },
        },
    },
    'train_dataloader': {
        'class': DataLoader,
        'args': {
            'batch_size': 4,
            'num_workers': 4,
            'collate_fn': {
                'class': ChangeStarCollator,
                'args': {
                    'method': 'train',
                },
            },
        },
    },
    'criterion': {
        'class': ChangeStarCriterion,
        'args': {},
    },
    'val_dataset': None,
    'val_dataloader': None,
    'test_dataset': None,
    'test_dataloader': None,
    'metric': {
        'class': ChangeStarMetric,
        'args': {},
    },
    'model': {
        'class': ChangeStar,
        'args': {
            'encoder': {
                'class': Resnet50WithFPN,
                'args': {},
            },
            'change_decoder_cfg': {
                'in_channels': 512,
                'mid_channels': 16,
                'out_channels': 2,
                'drop_rate': 0.2,
                'scale_factor': 4.0,
                'num_convs': 4,
            },
            'semantic_decoder_cfg': {
                'in_channels': 256,
                'out_channels': 5,
                'scale_factor': 4.0,
            },
        },
    },
    'optimizer': {
        'class': SingleTaskOptimizer,
        'args': {
            'optimizer_config': {
                'class': SGD,
                'args': {
                    'lr': 0.001,
                    'momentum': 0.9,
                    'weight_decay': 0.0001,
                },
            },
        },
    },
    'scheduler': {
        'class': PolynomialLR,
        'args': {
            'optimizer': None,
            'total_iters': None,
            'power': 0.9,
        },
    },
    'val_datasets': [{
    'class': xView2Dataset,
    'args': {
        'data_root': './data/datasets/soft_links/xView2',
        'split': 'test',
        'transforms_cfg': {
            'class': Compose,
            'args': {
                'transforms': [(
    {
            'class': RandomCrop,
            'args': {
                'size': (224, 224),
            },
        },
    [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'lbl_1'), ('labels', 'lbl_2')]
)],
            },
        },
    },
}, {
    'class': AirChangeDataset,
    'args': {
        'data_root': './data/datasets/soft_links/AirChange',
        'split': 'test',
        'transforms_cfg': {
            'class': Compose,
            'args': {
                'transforms': [(
    {
            'class': RandomCrop,
            'args': {
                'size': (112, 112),
                'resize': (224, 224),
                'interpolation': None,
            },
        },
    [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'change_map')]
)],
            },
        },
    },
}, {
    'class': CDDDataset,
    'args': {
        'data_root': './data/datasets/soft_links/CDD',
        'split': 'val',
        'transforms_cfg': {
            'class': Compose,
            'args': {
                'transforms': [(
    {
            'class': RandomCrop,
            'args': {
                'size': (224, 224),
                'resize': None,
                'interpolation': None,
            },
        },
    [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'change_map')]
)],
            },
        },
    },
}, {
    'class': LevirCdDataset,
    'args': {
        'data_root': './data/datasets/soft_links/LEVIR-CD',
        'split': 'val',
        'transforms_cfg': {
            'class': Compose,
            'args': {
                'transforms': [(
    {
            'class': RandomCrop,
            'args': {
                'size': (224, 224),
                'resize': None,
                'interpolation': None,
            },
        },
    [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'change_map')]
)],
            },
        },
    },
}, {
    'class': OSCDDataset,
    'args': {
        'data_root': './data/datasets/soft_links/OSCD',
        'split': 'test',
        'transforms_cfg': {
            'class': Compose,
            'args': {
                'transforms': [(
    {
            'class': RandomCrop,
            'args': {
                'size': (224, 224),
                'resize': None,
                'interpolation': None,
            },
        },
    [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'change_map')]
)],
            },
        },
        'bands': None,
    },
}, {
    'class': SYSU_CD_Dataset,
    'args': {
        'data_root': './data/datasets/soft_links/SYSU-CD',
        'split': 'val',
        'transforms_cfg': {
            'class': Compose,
            'args': {
                'transforms': [(
    {
            'class': RandomCrop,
            'args': {
                'size': (224, 224),
                'resize': None,
                'interpolation': None,
            },
        },
    [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'change_map')]
)],
            },
        },
    },
}],
    'val_dataloaders': [{
    'class': DataLoader,
    'args': {
        'batch_size': 1,
        'num_workers': 4,
        'collate_fn': {
            'class': ChangeStarCollator,
            'args': {
                'method': 'eval',
            },
        },
    },
}, {
    'class': DataLoader,
    'args': {
        'batch_size': 1,
        'num_workers': 4,
        'collate_fn': {
            'class': BaseCollator,
            'args': {
                'collators': {
                    'meta_info': {
                        'image_size': Tensor,
                        'crop_loc': Tensor,
                        'crop_size': Tensor,
                    },
                },
            },
        },
    },
}, {
    'class': DataLoader,
    'args': {
        'batch_size': 1,
        'num_workers': 4,
        'collate_fn': {
            'class': BaseCollator,
            'args': {
                'collators': {
                    'meta_info': {
                        'image_resolution': Tensor,
                    },
                },
            },
        },
    },
}, {
    'class': DataLoader,
    'args': {
        'batch_size': 1,
        'num_workers': 4,
        'collate_fn': {
            'class': BaseCollator,
            'args': {
                'collators': {
                    'meta_info': {
                        'image_resolution': Tensor,
                    },
                },
            },
        },
    },
}, {
    'class': DataLoader,
    'args': {
        'batch_size': 1,
        'num_workers': 4,
        'collate_fn': {
            'class': BaseCollator,
            'args': {
                'collators': {
                    'meta_info': {
                        'date_1': list,
                        'date_2': list,
                    },
                },
            },
        },
    },
}, {
    'class': DataLoader,
    'args': {
        'batch_size': 1,
        'num_workers': 4,
        'collate_fn': {
            'class': BaseCollator,
            'args': {
                'collators': {},
            },
        },
    },
}],
}
