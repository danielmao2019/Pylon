import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
os.chdir(REPO_ROOT)

from configs.benchmarks.point_cloud_registration.template_eval import (
    config as eval_template_config,
)
from configs.benchmarks.point_cloud_registration.template_train import (
    config as train_template_config,
)

from models.point_cloud_registration.classic import ICP, RANSAC_FPFH
from utils.automation.config_seeding import generate_seeded_configs
from utils.automation.config_to_file import add_heading
from utils.builders.builder import semideepcopy


def build_eval_config(dataset: str, model: str):
    """
    Build config for eval-only models (ICP, RANSAC_FPFH, TeaserPlusPlus).
    """
    # Start with eval template
    config = semideepcopy(eval_template_config)

    # Load dataset config
    if dataset == 'kitti':
        from configs.common.datasets.point_cloud_registration.eval.kitti_data_cfg import (
            data_cfg as dataset_cfg,
        )
    elif dataset == 'threedmatch':
        from configs.common.datasets.point_cloud_registration.eval.threedmatch_data_cfg import (
            data_cfg as dataset_cfg,
        )
    elif dataset == 'threedlomatch':
        from configs.common.datasets.point_cloud_registration.eval.threedlomatch_data_cfg import (
            data_cfg as dataset_cfg,
        )
    elif dataset == 'modelnet40':
        from configs.common.datasets.point_cloud_registration.eval.modelnet40_data_cfg import (
            data_cfg as dataset_cfg,
        )
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented for eval")

    # Load dataloader and metric configs
    from configs.common.dataloaders.point_cloud_registration.standard_dataloader_cfg import (
        dataloader_cfg,
    )
    from configs.common.metrics.point_cloud_registration.standard_metric_cfg import (
        metric_cfg,
    )

    # Update template with dataset, dataloader, and metric configs
    config.update(dataset_cfg)
    config['eval_dataloader'] = dataloader_cfg
    config['metric'] = metric_cfg

    # Model-specific config
    if model == 'TeaserPlusPlus':
        config['eval_n_jobs'] = 1
        from configs.common.models.point_cloud_registration.teaserplusplus_cfg import (
            model_cfg,
        )

        config['model'] = model_cfg
    else:
        # ICP or RANSAC_FPFH
        config['eval_n_jobs'] = 32
        model_class = ICP if model == 'ICP' else RANSAC_FPFH
        config['model'] = {'class': model_class, 'args': {}}

    return config


def build_training_config(dataset: str, model: str):
    """
    Build config for training models (GeoTransformer, OverlapPredator, PARENet, D3Feat).
    """
    # Start with training template
    config = semideepcopy(train_template_config)

    # Load dataset configs once for all models
    from configs.common.datasets.point_cloud_registration.train import (
        kitti_data_cfg,
        modelnet40_data_cfg,
        threedlomatch_data_cfg,
        threedmatch_data_cfg,
    )
    from configs.common.datasets.point_cloud_registration.val import (
        kitti_data_cfg as val_kitti,
    )
    from configs.common.datasets.point_cloud_registration.val import (
        modelnet40_data_cfg as val_modelnet40,
    )
    from configs.common.datasets.point_cloud_registration.val import (
        threedlomatch_data_cfg as val_threedlomatch,
    )
    from configs.common.datasets.point_cloud_registration.val import (
        threedmatch_data_cfg as val_threedmatch,
    )

    dataset_map = {
        'kitti': (kitti_data_cfg.data_cfg, val_kitti.data_cfg),
        'threedmatch': (threedmatch_data_cfg.data_cfg, val_threedmatch.data_cfg),
        'threedlomatch': (threedlomatch_data_cfg.data_cfg, val_threedlomatch.data_cfg),
        'modelnet40': (modelnet40_data_cfg.data_cfg, val_modelnet40.data_cfg),
    }

    # Get dataset configurations
    if dataset not in dataset_map:
        raise NotImplementedError(f"Dataset {dataset} not implemented")

    train_dataset_cfg, val_dataset_cfg = dataset_map[dataset]

    # Load all model-specific configs once
    from configs.common.criteria.point_cloud_registration import (
        d3feat_criterion_cfg,
        geotransformer_criterion_cfg,
        overlappredator_criterion_cfg,
        parenet_criterion_cfg,
    )
    from configs.common.dataloaders.point_cloud_registration import (
        d3feat_dataloader_cfg,
        geotransformer_dataloader_cfg,
        overlappredator_dataloader_cfg,
        parenet_dataloader_cfg,
    )
    from configs.common.metrics.point_cloud_registration import (
        d3feat_metric_cfg,
        geotransformer_metric_cfg,
        overlappredator_metric_cfg,
        parenet_metric_cfg,
    )
    from configs.common.models.point_cloud_registration import (
        geotransformer_cfg,
        overlappredator_cfg,
        parenet_cfg,
    )
    from configs.common.models.point_cloud_registration.d3feat import d3feat_model_cfg

    # Model configuration map
    model_map = {
        'GeoTransformer': {
            'train_dataloader_cfg': geotransformer_dataloader_cfg.train_dataloader_cfg,
            'val_dataloader_cfg': geotransformer_dataloader_cfg.val_dataloader_cfg,
            'model_cfg': geotransformer_cfg.model_cfg,
            'criterion_cfg': geotransformer_criterion_cfg.criterion_cfg,
            'metric_cfg': geotransformer_metric_cfg.metric_cfg,
        },
        'OverlapPredator': {
            'train_dataloader_cfg': overlappredator_dataloader_cfg.train_dataloader_cfg,
            'val_dataloader_cfg': overlappredator_dataloader_cfg.val_dataloader_cfg,
            'model_cfg': overlappredator_cfg.model_cfg,
            'criterion_cfg': overlappredator_criterion_cfg.criterion_cfg,
            'metric_cfg': overlappredator_metric_cfg.metric_cfg,
        },
        'PARENet': {
            'train_dataloader_cfg': parenet_dataloader_cfg.train_dataloader_cfg,
            'val_dataloader_cfg': parenet_dataloader_cfg.val_dataloader_cfg,
            'model_cfg': parenet_cfg.model_cfg,
            'criterion_cfg': parenet_criterion_cfg.criterion_cfg,
            'metric_cfg': parenet_metric_cfg.metric_cfg,
        },
        'D3Feat': {
            'train_dataloader_cfg': d3feat_dataloader_cfg.train_dataloader_cfg,
            'val_dataloader_cfg': d3feat_dataloader_cfg.val_dataloader_cfg,
            'model_cfg': d3feat_model_cfg.config,
            'criterion_cfg': d3feat_criterion_cfg.criterion_cfg,
            'metric_cfg': d3feat_metric_cfg.metric_cfg,
        },
    }

    # Get model configurations
    if model not in model_map:
        raise NotImplementedError(f"Model {model} not implemented")

    model_configs = model_map[model]
    train_dataloader_cfg = model_configs['train_dataloader_cfg']
    val_dataloader_cfg = model_configs['val_dataloader_cfg']
    model_cfg = model_configs['model_cfg']
    criterion_cfg = model_configs['criterion_cfg']
    metric_cfg = model_configs['metric_cfg']

    # Update template with all configs
    config.update(
        {
            'train_dataset': train_dataset_cfg['train_dataset'],
            'train_dataloader': train_dataloader_cfg,
            'criterion': criterion_cfg,
            'val_dataset': val_dataset_cfg['val_dataset'],
            'val_dataloader': val_dataloader_cfg,
            'metric': metric_cfg,
            'model': model_cfg,
        }
    )

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
        base_config=config, base_seed=relpath, base_work_dir=work_dir
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
            'modelnet40',
            'threedmatch',
            'threedlomatch',
        ],
        [
            'ICP',
            'RANSAC_FPFH',
            'TeaserPlusPlus',
            'GeoTransformer',
            'OverlapPredator',
            'PARENet',
            'D3Feat',
        ],
    )

    # Generate all configs
    for dataset, model in standard_combinations:
        main(dataset, model)
