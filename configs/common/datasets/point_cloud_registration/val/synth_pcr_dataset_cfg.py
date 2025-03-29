import data
import metrics


config = {
    'val_dataset': {
        'class': data.datasets.SynthPCRDataset,
        'args': {
            'data_root': './data/datasets/soft_links/ivision-pcr-data',
            'split': 'val',
            'rot_mag': 45.0,
            'trans_mag': 0.5,
            'voxel_size': 10.0,
        },
    },
    'val_dataloader': {
        'class': data.dataloaders.GeoTransformerDataloader,
        'args': {
            'batch_size': 1,
            'num_workers': 4,
            'num_stages': 4,
            'voxel_size': 1.0,
            'search_radius': 1.0,
        },
    },
    'metric': {
        'class': metrics.vision_3d.point_cloud_registration.GeoTransformerMetric,
        'args': {
            'eval': {
                'acceptance_overlap': 0.0,
                'acceptance_radius': 0.1,
                'inlier_ratio_threshold': 0.05,
                'rmse_threshold': 0.2,
                'rre_threshold': 15.0,
                'rte_threshold': 0.3,
            },
        },
    },
}
