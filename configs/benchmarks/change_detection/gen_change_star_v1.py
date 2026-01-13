import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
os.chdir(REPO_ROOT)

from configs.benchmarks.change_detection.template import config as template_config
from runners.trainers import MultiValDatasetTrainer
from utils.automation.config_seeding import generate_seeded_configs
from utils.automation.config_to_file import add_heading
from utils.builders.builder import semideepcopy


def build_config():
    """
    Build config for Change Star V1 model.
    """
    # Start with template
    config = semideepcopy(template_config)

    # Set runner
    config['runner'] = MultiValDatasetTrainer

    # Load dataset-specific configs
    from configs.common.datasets.change_detection.train.change_star_v1_xview2 import (
        data_cfg as train_dataset_config,
    )
    from configs.common.datasets.change_detection.val.change_star_v1_xview2 import (
        data_cfg as val_dataset_config,
    )

    # Update template with dataset configs
    config.update(train_dataset_config)
    config.update(val_dataset_config)

    # Add model config
    from configs.common.models.change_detection.change_star import model_config

    config['model'] = model_config

    return config


def generate_configs() -> None:
    """Generate config files for Change Star V1."""

    # Build config
    config = build_config()

    # Generate seeded configs
    relpath = os.path.join("benchmarks", "change_detection", "change_star_v1")
    work_dir = os.path.join("./logs", relpath, "xview2")

    # Generate seeded configs using the new dictionary-based approach
    seeded_configs = generate_seeded_configs(
        base_config=config, base_seed=relpath, base_work_dir=work_dir
    )

    # Add heading and save to disk
    generator_path = "./configs/benchmarks/change_detection/gen_change_star_v1.py"
    os.makedirs(os.path.join("./configs", relpath), exist_ok=True)
    for idx, seeded_config in enumerate(seeded_configs):
        # Add auto-generated header
        final_config = add_heading(seeded_config, generator_path)

        output_path = os.path.join("./configs", relpath, f"xview2_run_{idx}.py")
        with open(output_path, mode='w') as f:
            f.write(final_config)


def main() -> None:
    """Generate config files."""
    generate_configs()


if __name__ == "__main__":
    main()
