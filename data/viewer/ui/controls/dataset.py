"""Utility functions for the dataset viewer."""
import os
import importlib.util
import json
import random
import numpy as np
import torch
import traceback
from pathlib import Path


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


def create_dataset_selector(available_datasets):
    """
    Create a dataset selector dropdown.
    
    Args:
        available_datasets: Dictionary of available datasets
        
    Returns:
        html.Div containing the dataset selector
    """
    return html.Div([
        html.Label("Select Dataset:"),
        dcc.Dropdown(
            id='dataset-dropdown',
            options=[{'label': name, 'value': name} for name in sorted(available_datasets.keys())],
            value=None,
            style={'width': '100%'}
        )
    ], style={'width': '70%', 'display': 'inline-block', 'vertical-align': 'top'})


def create_reload_button():
    """
    Create a button to reload datasets.
    
    Returns:
        html.Div containing the reload button
    """
    return html.Div([
        html.Button(
            "Reload Datasets",
            id='reload-button',
            style={
                'background-color': '#007bff',
                'color': 'white',
                'border': 'none',
                'padding': '10px 15px',
                'cursor': 'pointer',
                'border-radius': '5px',
                'margin-top': '20px'
            }
        ),
    ], style={'width': '30%', 'display': 'inline-block', 'text-align': 'right'})

