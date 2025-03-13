"""Utility functions for the dataset viewer."""
import os
import importlib.util
import json
import random
import numpy as np
import torch
import traceback
from pathlib import Path

def format_value(value):
    """Format a value for display in error messages."""
    if isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(value, indent=2, default=str)
        except:
            return str(value)
    elif isinstance(value, torch.Tensor):
        return f"Tensor(shape={list(value.shape)}, dtype={value.dtype})"
    else:
        return str(value)

def get_available_datasets(config_dir=None):
    """Get a list of all available dataset configurations."""
    # If no config_dir is provided, use the default location
    if config_dir is None:
        # Adjust the path to be relative to the repository root
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        config_dir = os.path.join(repo_root, "configs/common/datasets/change_detection/train")
    
    if not os.path.exists(config_dir):
        print(f"Warning: Dataset directory not found at {config_dir}")
        return {}
        
    dataset_configs = {}
    
    for file in os.listdir(config_dir):
        if file.endswith('.py') and not file.startswith('_'):
            dataset_name = file[:-3]  # Remove .py extension
            try:
                # Try to import the config to ensure it's valid
                spec = importlib.util.spec_from_file_location(
                    f"configs.common.datasets.change_detection.train.{dataset_name}", 
                    os.path.join(config_dir, file)
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                if hasattr(module, 'config'):
                    # Add to the list of valid datasets
                    dataset_configs[dataset_name] = module.config
            except Exception as e:
                print(f"Error loading dataset config {dataset_name}: {e}")
    
    return dataset_configs
