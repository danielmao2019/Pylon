#!/usr/bin/env python3
"""Test script for eval viewer app that creates dummy data and launches the viewer.

This script creates temporary directories with dummy BaseTrainer and BaseEvaluator results,
then launches the eval viewer app to test the mixed result type functionality.

Usage:
    python tests/runners/eval_viewer/test_eval_viewer_app.py
"""
from typing import Dict, Any, List
import os
import sys
import shutil
import json
import numpy as np
import torch
import jsbeautifier

# Add project root to path
project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(project_root)

from runners.eval_viewer.app import run_app


def create_dummy_metric_scores(num_datapoints: int, epoch: int = 0, max_epochs: int = 5) -> Dict[str, Any]:
    """Create dummy metric scores in the format expected by the eval viewer.

    Args:
        num_datapoints: Number of datapoints in the dataset
        epoch: Current epoch (for score progression)
        max_epochs: Maximum number of epochs (for score progression)

    Returns:
        Dictionary with 'aggregated' and 'per_datapoint' scores
    """
    # Progress factor: scores increase from 0 to 1 over epochs
    progress = epoch / max(max_epochs - 1, 1)  # Avoid division by zero
    base_score = progress * 0.8 + 0.1  # Range from 0.1 to 0.9

    aggregated_scores = {}
    per_datapoint_scores = {}

    # Single-valued metrics
    single_metrics = ['metric1', 'metric2']
    for metric_name in single_metrics:
        # Generate per-datapoint scores with some random variation around base_score
        noise = np.random.uniform(-0.1, 0.1, num_datapoints)
        scores = np.clip(base_score + noise, 0.0, 1.0).tolist()
        per_datapoint_scores[metric_name] = scores

        # Calculate aggregated score as mean
        aggregated_scores[metric_name] = float(np.mean(scores))

    # Multi-valued metric (simulating class-wise scores)
    num_classes = 3
    class_scores = []
    for class_idx in range(num_classes):
        # Each class has slightly different performance
        class_bias = (class_idx - 1) * 0.05  # -0.05, 0, +0.05
        class_base = base_score + class_bias

        # Generate per-datapoint scores for this class
        noise = np.random.uniform(-0.1, 0.1, num_datapoints)
        scores = np.clip(class_base + noise, 0.0, 1.0).tolist()
        class_scores.append(scores)

    # Store multi-valued metric
    per_datapoint_scores['metric3'] = [[class_scores[i][dp] for i in range(num_classes)]
                                       for dp in range(num_datapoints)]
    aggregated_scores['metric3'] = [float(np.mean([class_scores[i][dp] for dp in range(num_datapoints)]))
                                   for i in range(num_classes)]

    return {
        'aggregated': aggregated_scores,
        'per_datapoint': per_datapoint_scores
    }


def create_trainer_log_dir(base_dir: str, run_name: str, num_epochs: int = 5, num_datapoints: int = 100) -> str:
    """Create a dummy BaseTrainer log directory structure.

    Args:
        base_dir: Base directory to create the log structure in
        run_name: Name of the run (e.g., 'DSAMNet_run_0')
        num_epochs: Number of epochs to simulate
        num_datapoints: Number of datapoints in the dataset

    Returns:
        Path to the created log directory
    """
    log_dir = os.path.join(base_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)

    # Create config.json
    config = {
        'model': {'class': 'Model1', 'args': {}},
        'val_dataset': {
            'class': 'COCOStuff164KDataset',
            'args': {
                'data_root': './data/datasets/soft_links/COCOStuff164K',
                'split': 'val2017',
                'semantic_granularity': 'coarse',
            }
        },
        'val_dataloader': {'class': 'DataLoader', 'args': {'batch_size': 8}},
        'metric': {'class': 'TestMetric', 'args': {}},
        'epochs': num_epochs,
        'seed': 42
    }

    with open(os.path.join(log_dir, "config.json"), 'w') as f:
        f.write(jsbeautifier.beautify(json.dumps(config), jsbeautifier.default_options()))

    # Create epoch directories with validation scores
    for epoch in range(num_epochs):
        epoch_dir = os.path.join(log_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)

        # Create validation_scores.json for this epoch
        scores = create_dummy_metric_scores(num_datapoints, epoch=epoch, max_epochs=num_epochs)

        with open(os.path.join(epoch_dir, "validation_scores.json"), 'w') as f:
            json.dump(scores, f, indent=2)

        # Create dummy checkpoint.pt and other required files
        checkpoint = {
            'model_state_dict': {},
            'optimizer_state_dict': {},
            'scheduler_state_dict': {}
        }
        torch.save(checkpoint, os.path.join(epoch_dir, "checkpoint.pt"))

        # Create training_losses.pt
        training_losses = {
            'total_loss': [0.5 * (1.1 ** -epoch) for _ in range(10)],  # Decreasing loss
            'ce_loss': [0.3 * (1.1 ** -epoch) for _ in range(10)]
        }
        torch.save(training_losses, os.path.join(epoch_dir, "training_losses.pt"))

        # Create optimizer_buffer.json
        optimizer_buffer = {
            'learning_rate': [0.001 * (0.9 ** epoch) for _ in range(10)]  # Decreasing LR
        }
        with open(os.path.join(epoch_dir, "optimizer_buffer.json"), 'w') as f:
            json.dump(optimizer_buffer, f, indent=2)

    return log_dir


def create_evaluator_log_dir(base_dir: str, run_name: str, num_datapoints: int = 100) -> str:
    """Create a dummy BaseEvaluator log directory structure.

    Args:
        base_dir: Base directory to create the log structure in
        run_name: Name of the run (e.g., 'DSAMNet_evaluation')
        num_datapoints: Number of datapoints in the dataset

    Returns:
        Path to the created log directory
    """
    log_dir = os.path.join(base_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)

    # Create config.json
    config = {
        'model': {'class': 'Model1', 'args': {}},
        'eval_dataset': {
            'class': 'COCOStuff164KDataset',
            'args': {
                'data_root': './data/datasets/soft_links/COCOStuff164K',
                'split': 'val2017',
                'semantic_granularity': 'coarse',
            }
        },
        'eval_dataloader': {'class': 'DataLoader', 'args': {'batch_size': 8}},
        'metric': {'class': 'TestMetric', 'args': {}},
        'seed': 42
    }

    with open(os.path.join(log_dir, "config.json"), 'w') as f:
        f.write(jsbeautifier.beautify(json.dumps(config), jsbeautifier.default_options()))

    # Create evaluation_scores.json directly in the log directory
    # For evaluator, use high epoch value to simulate well-trained model
    scores = create_dummy_metric_scores(num_datapoints, epoch=4, max_epochs=5)

    with open(os.path.join(log_dir, "evaluation_scores.json"), 'w') as f:
        json.dump(scores, f, indent=2)

    return log_dir


def create_config_files():
    """Create config files in the new folder structure."""
    # Create config directory structure
    config_dir = os.path.join(".", "configs", "tests", "runners", "eval_viewer", "semantic_segmentation", "coco_stuff_164k")
    os.makedirs(config_dir, exist_ok=True)
    
    # Config template for BaseTrainer runs
    trainer_config_template = """\"\"\"Configuration for {run_name}.\"\"\"

from data.datasets import COCOStuff164KDataset
import torch

config = {{
    'model': {{'class': '{model_class}', 'args': {{}}}},
    'val_dataset': {{
        'class': COCOStuff164KDataset,
        'args': {{
            'data_root': './data/datasets/soft_links/COCOStuff164K',
            'split': 'val2017',
            'semantic_granularity': 'coarse',
        }}
    }},
    'val_dataloader': {{
        'class': torch.utils.data.DataLoader,
        'args': {{'batch_size': 8}}
    }},
    'metric': {{'class': 'TestMetric', 'args': {{}}}},
    'epochs': {epochs},
    'seed': 42
}}
"""
    
    # Config template for BaseEvaluator runs
    evaluator_config_template = """\"\"\"Configuration for {run_name}.\"\"\"

from data.datasets import COCOStuff164KDataset
import torch

config = {{
    'model': {{'class': '{model_class}', 'args': {{}}}},
    'eval_dataset': {{
        'class': COCOStuff164KDataset,
        'args': {{
            'data_root': './data/datasets/soft_links/COCOStuff164K',
            'split': 'val2017',
            'semantic_granularity': 'coarse',
        }}
    }},
    'eval_dataloader': {{
        'class': torch.utils.data.DataLoader,
        'args': {{'batch_size': 8}}
    }},
    'metric': {{'class': 'TestMetric', 'args': {{}}}},
    'seed': 42
}}
"""
    
    # Create trainer configs
    trainer_configs = [
        ("model1_run_0", "Model1", 5),
        ("model2_run_0", "Model2", 3),
        ("model3_run_0", "Model3", 4)
    ]
    
    for run_name, model_class, epochs in trainer_configs:
        config_content = trainer_config_template.format(
            run_name=run_name,
            model_class=model_class,
            epochs=epochs
        )
        config_path = os.path.join(config_dir, f"{run_name}.py")
        with open(config_path, 'w') as f:
            f.write(config_content)
        print(f"Created config file: {config_path}")
    
    # Create evaluator configs
    evaluator_configs = [
        ("model1_evaluation", "Model1"),
        ("model2_evaluation", "Model2")
    ]
    
    for run_name, model_class in evaluator_configs:
        config_content = evaluator_config_template.format(
            run_name=run_name,
            model_class=model_class
        )
        config_path = os.path.join(config_dir, f"{run_name}.py")
        with open(config_path, 'w') as f:
            f.write(config_content)
        print(f"Created config file: {config_path}")


def create_dummy_log_dirs() -> List[str]:
    """Create dummy log directories for both BaseTrainer and BaseEvaluator results.

    Returns:
        List of paths to the created log directories
    """
    # Create base logs directory structure in project directory
    logs_dir = os.path.join(".", "logs", "tests", "runners", "eval_viewer", "semantic_segmentation", "coco_stuff_164k")
    os.makedirs(logs_dir, exist_ok=True)

    log_dirs = []

    # Create BaseTrainer results (3 different runs with different number of epochs)
    trainer_runs = [
        ("model1_run_0", 5),
        ("model2_run_0", 3),
        ("model3_run_0", 4)
    ]

    for run_name, num_epochs in trainer_runs:
        log_dir = create_trainer_log_dir(logs_dir, run_name, num_epochs=num_epochs, num_datapoints=100)
        log_dirs.append(log_dir)
        print(f"Created BaseTrainer log directory: {log_dir}")

    # Create BaseEvaluator results (2 evaluation runs)
    evaluator_runs = [
        "model1_evaluation",
        "model2_evaluation"
    ]

    for run_name in evaluator_runs:
        log_dir = create_evaluator_log_dir(logs_dir, run_name, num_datapoints=100)
        log_dirs.append(log_dir)
        print(f"Created BaseEvaluator log directory: {log_dir}")

    # Create corresponding config files
    create_config_files()

    return log_dirs


def main():
    """Main function to create dummy data and launch the eval viewer app."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test script for eval viewer app with dummy data")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the server on (default: 8050)")
    args = parser.parse_args()

    try:
        # Create dummy log directories in project directory
        log_dirs = create_dummy_log_dirs()

        print(f"\nCreated {len(log_dirs)} log directories:")
        for log_dir in log_dirs:
            print(f"  - {log_dir}")

        print(f"\nStarting eval viewer app with mixed BaseTrainer and BaseEvaluator results...")
        print(f"The app will show:")
        print(f"  - BaseTrainer results: respond to epoch slider changes")
        print(f"  - BaseEvaluator results: static display (ignore epoch slider)")
        print(f"Access the app at: http://localhost:{args.port}")
        print(f"\nPress Ctrl+C to stop the app and clean up test directories.")

        # Launch the eval viewer app
        run_app(log_dirs=log_dirs, force_reload=True, debug=True, port=args.port)

    except KeyboardInterrupt:
        print(f"\nStopping app...")
    finally:
        # Clean up test log directories
        test_logs_dir = "./logs/tests/runners/eval_viewer"
        if os.path.exists(test_logs_dir):
            print(f"Cleaning up test log directories: {test_logs_dir}")
            shutil.rmtree(test_logs_dir)
        
        # Clean up test config directories
        test_configs_dir = "./configs/tests/runners/eval_viewer"
        if os.path.exists(test_configs_dir):
            print(f"Cleaning up test config directories: {test_configs_dir}")
            shutil.rmtree(test_configs_dir)
        
        print("Done!")


if __name__ == "__main__":
    main()
