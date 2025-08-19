import os
import sys

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../.."))
sys.path.insert(0, project_root)
os.chdir(project_root)

from utils.automation.config_to_file import add_heading
from utils.automation.config_seeding import generate_seeded_configs
from utils.builders.builder import semideepcopy

# Import all necessary classes
from models.point_cloud_registration.classic import ICP, RANSAC_FPFH

# Load template configs
from configs.benchmarks.point_cloud_registration.template_eval import config as eval_template_config
from configs.benchmarks.point_cloud_registration.template_train import config as train_template_config
from configs.benchmarks.point_cloud_registration.template_buffer import config as buffer_template_config


def build_eval_config(dataset: str, model: str):
    """
    Build config for eval-only models (ICP, RANSAC_FPFH, TeaserPlusPlus).
    """
    # Determine dataset name and overlap
    if dataset.startswith('single_temporal_pcr') or dataset.startswith('bi_temporal_pcr'):
        overlap = float(dataset.split('_')[-1])
        dataset_name = '_'.join(dataset.split('_')[:-1])
    else:
        overlap = None
        dataset_name = dataset

    # Start with eval template
    config = semideepcopy(eval_template_config)

    # Load dataset-specific eval data config
    if dataset_name == 'kitti':
        from configs.common.datasets.point_cloud_registration.eval.kitti_data_cfg import data_cfg as eval_data_cfg
    elif dataset_name == 'single_temporal_pcr':
        from configs.common.datasets.point_cloud_registration.eval.single_temporal_pcr_data_cfg import data_cfg as eval_data_cfg
    elif dataset_name == 'bi_temporal_pcr':
        from configs.common.datasets.point_cloud_registration.eval.bi_temporal_pcr_data_cfg import data_cfg as eval_data_cfg
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented for eval")

    # Deep copy configs and set overlap if needed
    if overlap is not None:
        eval_data_cfg = semideepcopy(eval_data_cfg)
        eval_data_cfg['eval_dataset']['args']['overlap'] = overlap

    # Update template with dataset config
    config.update(eval_data_cfg)

    # Model-specific config
    if model == 'TeaserPlusPlus':
        config['eval_n_jobs'] = 1
        from configs.common.models.point_cloud_registration.teaserplusplus_cfg import model_cfg
        config['model'] = model_cfg
    else:
        # ICP or RANSAC_FPFH
        config['eval_n_jobs'] = 32
        model_class = ICP if model == 'ICP' else RANSAC_FPFH
        config['model'] = {'class': model_class, 'args': {}}

    return config


def build_training_config(dataset: str, model: str):
    """
    Build config for training models (GeoTransformer, OverlapPredator, BUFFER).
    """
    # Determine dataset name and overlap
    if dataset.startswith('single_temporal_pcr') or dataset.startswith('bi_temporal_pcr'):
        overlap = float(dataset.split('_')[-1])
        dataset_name = '_'.join(dataset.split('_')[:-1])
    else:
        overlap = None
        dataset_name = dataset

    if model == 'BUFFER':
        # Start with BUFFER template (list of stage configs)
        config = semideepcopy(buffer_template_config)
        # Note: BUFFER template already contains all necessary configuration
        # No additional customization needed for different datasets
        return config

    else:
        # Regular training models (GeoTransformer, OverlapPredator)
        # Start with training template
        config = semideepcopy(train_template_config)

        # Import train data config
        if model == 'GeoTransformer':
            if dataset_name == 'kitti':
                # For kitti, use general data config since no model-specific ones exist
                from configs.common.datasets.point_cloud_registration.train.kitti_data_cfg import data_cfg as train_data_cfg
                from configs.common.datasets.point_cloud_registration.val.kitti_data_cfg import data_cfg as val_data_cfg
            elif dataset_name == 'single_temporal_pcr':
                from configs.common.datasets.point_cloud_registration.train.geotransformer_single_temporal_pcr_data_cfg import data_cfg as train_data_cfg
                from configs.common.datasets.point_cloud_registration.val.geotransformer_single_temporal_pcr_data_cfg import data_cfg as val_data_cfg
            else:  # bi_temporal_pcr
                from configs.common.datasets.point_cloud_registration.train.geotransformer_bi_temporal_pcr_data_cfg import data_cfg as train_data_cfg
                from configs.common.datasets.point_cloud_registration.val.geotransformer_bi_temporal_pcr_data_cfg import data_cfg as val_data_cfg

            from configs.common.models.point_cloud_registration.geotransformer_cfg import model_cfg
            from configs.common.criteria.point_cloud_registration.geotransformer_criterion_cfg import criterion_cfg
            from configs.common.metrics.point_cloud_registration.geotransformer_metric_cfg import metric_cfg

        elif model == 'OverlapPredator':
            if dataset_name == 'kitti':
                # For kitti, use general data config since no model-specific ones exist
                from configs.common.datasets.point_cloud_registration.train.kitti_data_cfg import data_cfg as train_data_cfg
                from configs.common.datasets.point_cloud_registration.val.kitti_data_cfg import data_cfg as val_data_cfg
            elif dataset_name == 'single_temporal_pcr':
                from configs.common.datasets.point_cloud_registration.train.overlappredator_single_temporal_pcr_data_cfg import data_cfg as train_data_cfg
                from configs.common.datasets.point_cloud_registration.val.overlappredator_single_temporal_pcr_data_cfg import data_cfg as val_data_cfg
            else:  # bi_temporal_pcr
                from configs.common.datasets.point_cloud_registration.train.overlappredator_bi_temporal_pcr_data_cfg import data_cfg as train_data_cfg
                from configs.common.datasets.point_cloud_registration.val.overlappredator_bi_temporal_pcr_data_cfg import data_cfg as val_data_cfg

            from configs.common.models.point_cloud_registration.overlappredator_cfg import model_cfg
            from configs.common.criteria.point_cloud_registration.overlappredator_criterion_cfg import criterion_cfg
            from configs.common.metrics.point_cloud_registration.overlappredator_metric_cfg import metric_cfg
        
        elif model == 'GMCNet':
            # GMCNet only works with ModelNet40 dataset
            if dataset_name not in ['modelnet40']:
                raise NotImplementedError(f"GMCNet is only compatible with modelnet40 dataset, not {dataset_name}")
                
            from configs.common.datasets.point_cloud_registration.train.gmcnet_modelnet40_data_cfg import data_cfg as train_data_cfg
            from configs.common.datasets.point_cloud_registration.val.gmcnet_modelnet40_data_cfg import data_cfg as val_data_cfg

            from configs.common.models.point_cloud_registration.gmcnet_cfg import model_cfg
            from criteria.vision_3d.point_cloud_registration.gmcnet_criterion import GMCNetCriterion
            from metrics.vision_3d.point_cloud_registration.gmcnet_metric import GMCNetMetric
            
            criterion_cfg = {'class': GMCNetCriterion, 'args': {}}
            metric_cfg = {'class': GMCNetMetric, 'args': {}}
        
        elif model == 'D3Feat':
            # D3Feat only works with ThreeDMatch dataset
            if dataset_name != 'threedmatch':
                raise NotImplementedError(f"D3Feat is only compatible with threedmatch dataset, not {dataset_name}")
                
            from configs.common.datasets.point_cloud_registration.train.d3feat_threedmatch_data_cfg import data_cfg as train_data_cfg
            from configs.common.datasets.point_cloud_registration.val.d3feat_threedmatch_data_cfg import data_cfg as val_data_cfg

            from configs.common.models.point_cloud_registration.d3feat.d3feat_model_cfg import config as model_cfg
            from configs.common.criteria.point_cloud_registration.d3feat_criterion_cfg import criterion_cfg
            from configs.common.metrics.point_cloud_registration.d3feat_metric_cfg import metric_cfg
            
        elif model == 'PARENet':
            if dataset_name == 'kitti':
                # Use PARENet-specific data config with collate_fn
                from configs.common.datasets.point_cloud_registration.train.parenet_kitti_data_cfg import data_cfg as train_data_cfg
                from configs.common.datasets.point_cloud_registration.val.parenet_kitti_data_cfg import data_cfg as val_data_cfg
            elif dataset_name == 'single_temporal_pcr':
                from configs.common.datasets.point_cloud_registration.train.geotransformer_single_temporal_pcr_data_cfg import data_cfg as train_data_cfg
                from configs.common.datasets.point_cloud_registration.val.geotransformer_single_temporal_pcr_data_cfg import data_cfg as val_data_cfg
            else:  # bi_temporal_pcr
                from configs.common.datasets.point_cloud_registration.train.geotransformer_bi_temporal_pcr_data_cfg import data_cfg as train_data_cfg
                from configs.common.datasets.point_cloud_registration.val.geotransformer_bi_temporal_pcr_data_cfg import data_cfg as val_data_cfg

            from configs.common.models.point_cloud_registration.parenet_cfg import model_cfg
            from configs.common.criteria.point_cloud_registration.parenet_criterion_cfg import criterion_cfg
            from configs.common.metrics.point_cloud_registration.parenet_metric_cfg import metric_cfg
        else:
            raise NotImplementedError(f"Model {model} not implemented")

        # Deep copy configs and set overlap if needed
        if overlap is not None:
            train_data = semideepcopy(train_data_cfg)
            val_data = semideepcopy(val_data_cfg)
            train_data['train_dataset']['args']['overlap'] = overlap
            val_data['val_dataset']['args']['overlap'] = overlap
        else:
            train_data = train_data_cfg
            val_data = val_data_cfg

        # Update template with dataset and model configs
        config.update({
            'train_dataset': train_data['train_dataset'],
            'train_dataloader': train_data['train_dataloader'],
            'criterion': criterion_cfg,
            'val_dataset': val_data['val_dataset'],
            'val_dataloader': val_data['val_dataloader'],
            'metric': metric_cfg,
            'model': model_cfg,
        })

        return config


def generate_configs(dataset: str, model: str) -> None:
    """Generate config files for a specific dataset and model combination."""

    # Build appropriate config based on model type
    if model in ['ICP', 'RANSAC_FPFH', 'TeaserPlusPlus']:
        config = build_eval_config(dataset, model)
    else:
        config = build_training_config(dataset, model)

    # Generate seeded configs
    relpath = os.path.join("benchmarks", "point_cloud_registration", dataset)
    work_dir = os.path.join("./logs", relpath, model)

    # Generate seeded configs using the new dictionary-based approach
    seeded_configs = generate_seeded_configs(
        base_config=config,
        base_seed=relpath,
        base_work_dir=work_dir
    )

    # Add heading and save to disk
    generator_path = "./configs/benchmarks/point_cloud_registration/gen.py"
    os.makedirs(os.path.join("./configs", relpath), exist_ok=True)
    for idx, seeded_config in enumerate(seeded_configs):
        # Add auto-generated header
        final_config = add_heading(seeded_config, generator_path)

        output_path = os.path.join("./configs", relpath, f"{model}_run_{idx}.py")
        with open(output_path, mode='w') as f:
            f.write(final_config)


def main(dataset: str, model: str) -> None:
    """Generate config file for a specific dataset and model combination."""
    generate_configs(dataset, model)


if __name__ == "__main__":
    import itertools
    
    # Standard datasets and models
    standard_combinations = itertools.product(
        [
            'kitti',
        ],
        [
            'ICP', 'RANSAC_FPFH', 'TeaserPlusPlus',
            'GeoTransformer', 'OverlapPredator', 'BUFFER', 'GMCNet', 'PARENet',
        ],
    )
    
    # D3Feat only works with ThreeDMatch
    d3feat_combinations = [('threedmatch', 'D3Feat')]
    
    # Generate all configs
    for dataset, model in itertools.chain(standard_combinations, d3feat_combinations):
        main(dataset, model)
