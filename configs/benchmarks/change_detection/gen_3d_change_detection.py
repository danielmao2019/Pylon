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
import torch
import optimizers
import data
import models
import criteria
import metrics
from runners.trainers import SupervisedSingleTaskTrainer

# Load template config
from configs.benchmarks.change_detection.template import config as template_config


def build_config(dataset: str, model: str):
    """
    Build config for 3D change detection models.
    """
    # Start with template
    config = semideepcopy(template_config)

    # Set runner
    config['runner'] = SupervisedSingleTaskTrainer

    # Load dataset-specific configs
    if dataset == "urb3dcd":
        from configs.common.datasets.change_detection.train.urb3dcd_data_cfg import data_cfg as train_data_cfg
        from configs.common.datasets.change_detection.val.urb3dcd_data_cfg import data_cfg as val_data_cfg
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented")

    # Update template with dataset configs
    config.update(train_data_cfg)
    config.update(val_data_cfg)

    # Model-specific configurations
    if model == "SiameseKPConv":
        from configs.common.models.change_detection.siamese_kpconv import model_config
        config['model'] = model_config

        # Special collator for point clouds
        if dataset == "urb3dcd":
            # Use specialized point cloud collator
            config['train_dataloader']['args']['collate_fn'] = {'class': data.collators.SiameseKPConvCollator, 'args': {}}
            config['val_dataloader']['args']['collate_fn'] = {'class': data.collators.SiameseKPConvCollator, 'args': {}}
    else:
        raise NotImplementedError(f"3D model {model} not supported!")

    return config


def generate_configs(dataset: str, model: str) -> None:
    """Generate config files for a specific dataset and model combination."""

    # Build config
    config = build_config(dataset, model)

    # Generate seeded configs
    relpath = os.path.join("benchmarks", "change_detection", dataset)
    work_dir = os.path.join("./logs", relpath, model)

    # Generate seeded configs using the new dictionary-based approach
    seeded_configs = generate_seeded_configs(
        base_config=config,
        base_seed=relpath,
        base_work_dir=work_dir
    )

    # Add heading and save to disk
    generator_path = "./configs/benchmarks/change_detection/gen_3d_change_detection.py"
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
    for dataset, model in itertools.product(
        ['urb3dcd'],
        [
            'SiameseKPConv',
        ],
    ):
        main(dataset, model)
