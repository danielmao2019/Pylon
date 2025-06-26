"""Dummy data generators for change detection model testing."""
from typing import Dict
import torch


def generate_change_detection_data(
    batch_size: int = 2,
    height: int = 224,
    width: int = 224,
    channels: int = 3,
    device: str = 'cpu'
) -> Dict[str, torch.Tensor]:
    """Generate dummy data for change detection models.
    
    Args:
        batch_size: Number of samples in batch
        height: Image height
        width: Image width  
        channels: Number of channels
        device: Device to place tensors on
        
    Returns:
        Dictionary with 'img_1' and 'img_2' keys
    """
    return {
        'img_1': torch.randn(batch_size, channels, height, width, dtype=torch.float32, device=device),
        'img_2': torch.randn(batch_size, channels, height, width, dtype=torch.float32, device=device)
    }


def generate_change_labels(
    batch_size: int = 2,
    height: int = 224,
    width: int = 224,
    device: str = 'cpu'
) -> torch.Tensor:
    """Generate dummy change detection labels.
    
    Args:
        batch_size: Number of samples in batch
        height: Label height
        width: Label width
        device: Device to place tensors on
        
        
    Returns:
        Binary change labels tensor (N, H, W) int64
    """
    return torch.randint(0, 2, (batch_size, height, width), dtype=torch.int64, device=device)


def generate_segmentation_labels(
    batch_size: int = 2,
    height: int = 224,
    width: int = 224,
    num_classes: int = 10,
    device: str = 'cpu'
) -> torch.Tensor:
    """Generate dummy segmentation labels.
    
    Args:
        batch_size: Number of samples in batch
        height: Label height
        width: Label width
        num_classes: Number of classes
        device: Device to place tensors on
        
    Returns:
        Segmentation labels tensor (N, H, W) int64
    """
    return torch.randint(0, num_classes, (batch_size, height, width), dtype=torch.int64, device=device)


def create_minimal_change_detection_input(device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """Create minimal valid input for testing edge cases.
    
    Args:
        device: Device to place tensors on
        
    Returns:
        Minimal valid input data
    """
    return generate_change_detection_data(
        batch_size=1, height=32, width=32, device=device
    )


def create_large_change_detection_input(device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """Create larger input for memory testing.
    
    Args:
        device: Device to place tensors on
        
    Returns:
        Large input data for memory testing
    """
    return generate_change_detection_data(
        batch_size=4, height=512, width=512, device=device
    )