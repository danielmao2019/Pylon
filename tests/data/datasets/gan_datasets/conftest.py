"""Shared fixtures and helper functions for GAN dataset tests."""

import pytest
import torch
from data.datasets import MNISTDataset  
from data.datasets.gan_datasets.gan_dataset import GANDataset
from utils.builders.builder import build_from_config


@pytest.fixture
def gan_dataset_config(request, mnist_data_root):
    """Fixture for creating a GANDataset config with parameterized settings."""
    split, device_str = request.param  # Unpack the test parameters
    latent_dim = 128
    
    # Device handling
    if device_str == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device(device_str)
    
    # Create source dataset config
    source_config = {
        'class': MNISTDataset,
        'args': {
            'data_root': mnist_data_root,
            'split': split,
            'device': torch.device('cpu')  # Source should always be CPU for BaseSyntheticDataset
        }
    }
    
    # Build source dataset
    source = build_from_config(source_config)
    
    return {
        'class': GANDataset,
        'args': {
            'source': source,
            'dataset_size': len(source),
            'latent_dim': latent_dim,
            'device': device
        }
    }