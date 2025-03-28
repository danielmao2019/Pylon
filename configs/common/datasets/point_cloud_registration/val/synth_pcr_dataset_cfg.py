import data
import metrics


config = {
    'val_dataset': {
        'class': data.datasets.SynthPCRDataset,
        'args': {
            'root_dir': './data/datasets/soft_links/ivision-pcr-data',
            'split': 'val',
            'rot_mag': 45.0,
            'trans_mag': 0.5,
        },
    },
    'val_dataloader': {
        'class': data.dataloaders.GeoTransformerDataloader,
        'args': {
            'batch_size': 1,
            'num_workers': 4,
            'num_stages': 4,
            'voxel_size': 0.025,
            'search_radius': 2.5 * 0.025,
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
