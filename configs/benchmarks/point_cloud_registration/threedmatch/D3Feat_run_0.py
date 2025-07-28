# This is a sample D3Feat config for ThreeDMatch dataset
# Generated config will follow this structure when C++ extensions are compiled

from models.point_cloud_registration.d3feat import D3FeatModel
from criteria.vision_3d.point_cloud_registration import CircleLoss  
from metrics.vision_3d.point_cloud_registration import D3FeatDescriptorMetric
from data.datasets.pcr_datasets.threedmatch_dataset import ThreeDMatchDataset

config = {
    'model': {
        'class': D3FeatModel,
        'args': {
            'num_layers': 5,
            'in_points_dim': 3,
            'first_features_dim': 128,
            'first_subsampling_dl': 0.03,
            'in_features_dim': 1,
            'conv_radius': 2.5,
            'deform_radius': 5.0,
        }
    },
    'criterion': {
        'class': CircleLoss,
        'args': {
            'pos_margin': 0.1,
            'neg_margin': 1.4,
            'safe_radius': 0.04,
        }
    },
    'metric': {
        'class': D3FeatDescriptorMetric,
        'args': {
            'distance_threshold': 0.1,
        }
    },
    'train_dataset': {
        'class': ThreeDMatchDataset,
        'args': {
            'data_root': './data/datasets/soft_links/threedmatch',
            'split': 'train',
        }
    }
}