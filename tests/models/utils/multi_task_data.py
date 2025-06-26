"""Dummy data generators for multi-task and general model testing."""
from typing import Dict, Tuple
import torch


def generate_multi_task_data(
    batch_size: int = 2,
    height: int = 224,
    width: int = 224,
    channels: int = 3,
    device: str = 'cpu'
) -> Dict[str, torch.Tensor]:
    """Generate dummy data for multi-task learning models.
    
    Args:
        batch_size: Number of samples in batch
        height: Image height
        width: Image width
        channels: Number of channels
        device: Device to place tensors on
        
    Returns:
        Dictionary with image data
    """
    return {
        'images': torch.randn(batch_size, channels, height, width, dtype=torch.float32, device=device)
    }


def generate_classification_labels(
    batch_size: int = 2,
    num_classes: int = 10,
    device: str = 'cpu'
) -> torch.Tensor:
    """Generate dummy classification labels.
    
    Args:
        batch_size: Number of samples in batch
        num_classes: Number of classes
        device: Device to place tensors on
        
    Returns:
        Classification labels tensor (N,) int64
    """
    return torch.randint(0, num_classes, (batch_size,), dtype=torch.int64, device=device)


def get_model_input_output_shapes(model_type: str) -> Dict[str, Tuple]:
    """Get expected input/output shapes for different model types.
    
    Args:
        model_type: Type of model ('change_detection', 'point_cloud', 'multi_task')
        
    Returns:
        Dictionary with input and output shape information
    """
    if model_type == 'change_detection':
        return {
            'input_shapes': {
                'img_1': (2, 3, 224, 224),
                'img_2': (2, 3, 224, 224)
            },
            'output_shapes': {
                'change_mask': (2, 224, 224),
                'semantic_seg': (2, 10, 224, 224)  # If multi-task
            }
        }
    elif model_type == 'point_cloud':
        return {
            'input_shapes': {
                'src_pc': {'pos': (2, 1024, 3), 'feat': (2, 1024, 32)},
                'tgt_pc': {'pos': (2, 1024, 3), 'feat': (2, 1024, 32)}
            },
            'output_shapes': {
                'transformation': (2, 4, 4),
                'correspondences': (2, 1024, 1024)  # If applicable
            }
        }
    elif model_type == 'multi_task':
        return {
            'input_shapes': {
                'images': (2, 3, 224, 224)
            },
            'output_shapes': {
                'segmentation': (2, 19, 224, 224),  # CityScapes classes
                'depth': (2, 1, 224, 224),
                'surface_normal': (2, 3, 224, 224)
            }
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_minimal_multi_task_input(device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """Create minimal valid input for testing edge cases.
    
    Args:
        device: Device to place tensors on
        
    Returns:
        Minimal valid input data
    """
    return generate_multi_task_data(
        batch_size=1, height=32, width=32, device=device
    )


def create_large_multi_task_input(device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """Create larger input for memory testing.
    
    Args:
        device: Device to place tensors on
        
    Returns:
        Large input data for memory testing
    """
    return generate_multi_task_data(
        batch_size=4, height=512, width=512, device=device
    )