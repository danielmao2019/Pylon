import pytest
import tempfile
import json
import numpy as np
import os
from pathlib import Path
from typing import Dict, Any, List


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for log files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_scores_dict():
    """Create a sample scores dictionary matching expected format."""
    return {
        'aggregated': {
            'metric1': 0.75,
            'metric2': 0.85,
            'multi_metric': [0.8, 0.7, 0.9]
        },
        'per_datapoint': {
            'metric1': [0.7, 0.8, 0.75, 0.72, 0.78],
            'metric2': [0.9, 0.8, 0.85, 0.87, 0.83],
            'multi_metric': [
                [0.8, 0.7, 0.9],
                [0.75, 0.72, 0.85],
                [0.82, 0.68, 0.92],
                [0.78, 0.74, 0.88],
                [0.85, 0.71, 0.91]
            ]
        }
    }


@pytest.fixture
def validation_scores_file(temp_log_dir, sample_scores_dict):
    """Create a temporary validation_scores.json file."""
    scores_file = os.path.join(temp_log_dir, "validation_scores.json")
    with open(scores_file, 'w') as f:
        json.dump(sample_scores_dict, f)
    return scores_file


@pytest.fixture
def sample_score_maps():
    """Create sample score maps for testing overlaid functionality."""
    # Create 3 score maps of size 3x3 with known values
    map1 = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, np.nan]
    ])

    map2 = np.array([
        [0.15, 0.25, 0.35],
        [0.45, 0.55, 0.65],
        [0.75, 0.85, np.nan]
    ])

    map3 = np.array([
        [0.05, 0.15, 0.25],
        [0.35, 0.45, 0.55],
        [0.65, 0.75, np.nan]
    ])

    return [map1, map2, map3]


@pytest.fixture
def trainer_log_structure(temp_log_dir, sample_scores_dict):
    """Create a BaseTrainer log directory structure."""
    # Create epoch directories with validation scores
    epoch_dirs = []
    for epoch in range(3):
        epoch_dir = os.path.join(temp_log_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)

        # Create validation_scores.json with slight variations per epoch
        scores = sample_scores_dict.copy()
        # Add slight progression to scores
        factor = 1.0 + epoch * 0.1
        scores['aggregated']['metric1'] *= factor
        scores['aggregated']['metric2'] *= factor

        scores_file = os.path.join(epoch_dir, "validation_scores.json")
        with open(scores_file, 'w') as f:
            json.dump(scores, f)

        epoch_dirs.append(epoch_dir)

    return epoch_dirs


@pytest.fixture
def evaluator_log_structure(temp_log_dir, sample_scores_dict):
    """Create a BaseEvaluator log directory structure."""
    # Create evaluation_scores.json directly in log directory
    scores_file = os.path.join(temp_log_dir, "evaluation_scores.json")
    with open(scores_file, 'w') as f:
        json.dump(sample_scores_dict, f)

    return temp_log_dir


@pytest.fixture
def mock_config_file(temp_log_dir):
    """Create a mock config file for testing."""
    # Create configs structure matching expected path
    config_dir = os.path.join(temp_log_dir, "configs")
    os.makedirs(config_dir, exist_ok=True)

    config_content = '''from data.datasets import COCOStuff164KDataset
import torch

config = {
    "val_dataset": {
        "class": COCOStuff164KDataset,
        "args": {
            "data_root": "./data/datasets/soft_links/COCOStuff164K",
            "split": "val2017",
            "semantic_granularity": "coarse"
        }
    },
    "val_dataloader": {
        "class": torch.utils.data.DataLoader,
        "args": {"batch_size": 8}
    }
}'''

    config_file = os.path.join(config_dir, "test_config.py")
    with open(config_file, 'w') as f:
        f.write(config_content)

    return config_file
