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
from runners import MultiValDatasetTrainer

# Load template config
from configs.benchmarks.change_detection.template import config as template_config


def build_config():
    """
    Build config for PPSL model.
    """
    # Start with template
    config = semideepcopy(template_config)
    
    # Set runner
    config['runner'] = MultiValDatasetTrainer
    
    # Load dataset-specific configs
    from configs.common.datasets.change_detection.train.ppsl_whu_bd import data_cfg as train_dataset_config
    from configs.common.datasets.change_detection.val.all_bi_temporal import data_cfg as val_dataset_config
    
    # Update template with dataset configs
    config.update(train_dataset_config)
    config.update(val_dataset_config)
    
    # Add model config
    from configs.common.models.change_detection.ppsl_model import model_config
    config['model'] = model_config
    
    return config


def generate_configs() -> None:
    """Generate config files for PPSL."""
    
    # Build config
    config = build_config()
    
    # Generate seeded configs
    relpath = os.path.join("benchmarks", "change_detection", "ppsl")
    work_dir = os.path.join("./logs", relpath, "whu_bd")
    
    # Generate seeded configs using the new dictionary-based approach
    seeded_configs = generate_seeded_configs(
        base_config=config,
        base_seed=relpath,
        base_work_dir=work_dir
    )
    
    # Add heading and save to disk
    generator_path = "./configs/benchmarks/change_detection/gen_ppsl.py"
    os.makedirs(os.path.join("./configs", relpath), exist_ok=True)
    for idx, seeded_config in enumerate(seeded_configs):
        # Add auto-generated header
        final_config = add_heading(seeded_config, generator_path)
        
        output_path = os.path.join("./configs", relpath, f"whu_bd_run_{idx}.py")
        with open(output_path, mode='w') as f:
            f.write(final_config)


def main() -> None:
    """Generate config files."""
    generate_configs()


if __name__ == "__main__":
    main()
