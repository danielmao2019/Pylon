"""
New seeding utilities that work with config dictionaries directly instead of string manipulation.
This replaces the old string-based approach in utils.automation.configs.
"""

from typing import Dict, List, Any, Optional, Union
import random
from utils.automation.config_to_file import dict_to_config_file
from utils.builders.builder import semideepcopy


def generate_seeded_configs(
    base_config: Union[Dict[str, Any], List[Dict[str, Any]]],
    base_seed: str,
    base_work_dir: Optional[str] = None,
    num_repetitions: Optional[int] = 3,
    ub: Optional[int] = 10**8-1,
) -> List[str]:
    """
    Generate seeded config files from a base config dictionary.

    Args:
        base_config: The base configuration dictionary
        base_seed: String used to seed the random number generator
        base_work_dir: Base work directory path (will append _run_{idx})
        num_repetitions: Number of repetitions to generate
        ub: Upper bound for random seed generation

    Returns:
        List of generated config file contents as strings
    """
    # Check if this is a list of configs (BUFFER multi-stage)
    if isinstance(base_config, list):
        return _generate_multistage_seeded_configs(
            base_config, base_seed, base_work_dir, num_repetitions, ub
        )
    # Check if this is an eval config or training config
    elif 'epochs' in base_config and base_config['epochs'] is not None:
        return _generate_train_seeded_configs(
            base_config, base_seed, base_work_dir, num_repetitions, ub
        )
    else:
        return _generate_eval_seeded_configs(
            base_config, base_seed, base_work_dir, ub
        )


def _generate_train_seeded_configs(
    base_config: Dict[str, Any],
    base_seed: str,
    base_work_dir: Optional[str] = None,
    num_repetitions: Optional[int] = 3,
    ub: Optional[int] = 10**8-1,
) -> List[str]:
    """Generate seeded configs for training models."""
    seeded_configs: List[str] = []
    epochs = base_config['epochs']

    for idx in range(num_repetitions):
        # Seed the random generator
        random.seed(base_seed + str(idx))

        # Deep copy the base config
        config = semideepcopy(base_config)

        # Generate and set seeds directly in the config dictionary
        config['init_seed'] = random.randint(0, ub)
        config['train_seeds'] = [random.randint(0, ub) for _ in range(epochs)]
        config['val_seeds'] = [random.randint(0, ub) for _ in range(epochs)]
        config['test_seed'] = random.randint(0, ub)

        # Set work directory
        if base_work_dir is not None:
            config['work_dir'] = f"{base_work_dir}_run_{idx}"

        # Generate the config file content
        config_content = dict_to_config_file(config=config)

        seeded_configs.append(config_content)

    return seeded_configs


def _generate_eval_seeded_configs(
    base_config: Dict[str, Any],
    base_seed: str,
    base_work_dir: Optional[str] = None,
    ub: Optional[int] = 10**8-1,
) -> List[str]:
    """Generate seeded configs for eval models."""
    # Seed the random generator
    random.seed(base_seed)

    # Deep copy the base config
    config = semideepcopy(base_config)

    # Generate and set seed directly in the config dictionary
    config['seed'] = random.randint(0, ub)

    # Set work directory
    if base_work_dir is not None:
        config['work_dir'] = f"{base_work_dir}_run_0"

    # Generate the config file content
    config_content = dict_to_config_file(config=config)

    return [config_content]


def _generate_multistage_seeded_configs(
    base_config: List[Dict[str, Any]],
    base_seed: str,
    base_work_dir: Optional[str] = None,
    num_repetitions: Optional[int] = 3,
    ub: Optional[int] = 10**8-1,
) -> List[str]:
    """Generate seeded configs for multi-stage models like BUFFER."""
    seeded_configs: List[str] = []

    # Get epochs from each stage (should be same for all stages)
    epochs_per_stage = [stage_config['epochs'] for stage_config in base_config]

    for idx in range(num_repetitions):
        # Seed the random generator
        random.seed(base_seed + str(idx))

        # Deep copy the base config list
        config = semideepcopy(base_config)

        # Generate different seeds for each stage (to match old behavior)
        init_seed_multi_stage = [random.randint(0, ub) for _ in range(len(epochs_per_stage))]
        train_seeds_multi_stage = [
            [random.randint(0, ub) for _ in range(num_epochs)]
            for num_epochs in epochs_per_stage
        ]
        val_seeds_multi_stage = [
            [random.randint(0, ub) for _ in range(num_epochs)]
            for num_epochs in epochs_per_stage
        ]
        test_seed_multi_stage = [random.randint(0, ub) for _ in range(len(epochs_per_stage))]

        # Apply different seeds to each stage
        for stage_idx, stage_config in enumerate(config):
            stage_config['init_seed'] = init_seed_multi_stage[stage_idx]
            stage_config['train_seeds'] = train_seeds_multi_stage[stage_idx]
            stage_config['val_seeds'] = val_seeds_multi_stage[stage_idx]
            stage_config['test_seed'] = test_seed_multi_stage[stage_idx]

            # Set work directory for each stage
            if base_work_dir is not None:
                stage_config['work_dir'] = f"{base_work_dir}_run_{idx}"

        # Generate the config file content
        config_content = dict_to_config_file(config=config)

        seeded_configs.append(config_content)

    return seeded_configs
