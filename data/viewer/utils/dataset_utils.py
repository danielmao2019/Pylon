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

def tensor_to_image(tensor):
    """Convert a PyTorch tensor to a displayable image."""
    img = tensor.cpu().numpy()
    img = (img-img.min())/(img.max()-img.min())
    if img.ndim == 2:  # Grayscale image
        return img
    elif img.ndim == 3:  # RGB image (C, H, W) -> (H, W, C)
        if img.shape[0] > 3:
            img = img[random.sample(range(img.shape[0]), 3), :, :]
        return np.transpose(img, (1, 2, 0))
    else:
        raise ValueError("Unsupported tensor shape for image conversion")

def is_point_cloud(data):
    """Check if the input data is a point cloud."""
    if isinstance(data, torch.Tensor):
        # Point cloud typically has shape (N, 3) or more
        return data.ndim == 2 and data.shape[1] >= 3
    return False

def is_3d_dataset(dataset, datapoint=None):
    """
    Detect if a dataset is 3D based on its class name or datapoint structure.
    """
    # Check class name indicators
    if 'Point' in dataset.__class__.__name__ or '3D' in dataset.__class__.__name__:
        return True
    
    # If we have a datapoint, check its structure
    if datapoint is not None:
        # Check for common 3D point cloud keys
        if 'pc_0' in datapoint['inputs'] or 'pc_1' in datapoint['inputs'] or 'point_cloud' in datapoint['inputs']:
            return True
            
        # Check tensor dimensions for potential point clouds
        for key, value in datapoint['inputs'].items():
            if isinstance(value, torch.Tensor) and len(value.shape) == 3 and value.shape[2] == 3:
                # This might be a point cloud with [N, 1, 3] or similar shape
                return True
                
    # Default to 2D if we can't determine
    return False

def get_point_cloud_stats(pc, change_map=None, class_names=None):
    """Get statistical information about a point cloud.

    Args:
        pc: Point cloud tensor of shape (N, 3+)
        change_map: Optional tensor with change classes for each point
        class_names: Optional dictionary mapping class IDs to class names

    Returns:
        Dictionary with point cloud statistics
    """
    if not isinstance(pc, torch.Tensor):
        return {}

    try:
        # Basic stats
        pc_np = pc.detach().cpu().numpy()
        stats = {
            "Total Points": len(pc_np),
            "Dimensions": pc_np.shape[1],
            "X Range": f"[{pc_np[:, 0].min():.2f}, {pc_np[:, 0].max():.2f}]",
            "Y Range": f"[{pc_np[:, 1].min():.2f}, {pc_np[:, 1].max():.2f}]",
            "Z Range": f"[{pc_np[:, 2].min():.2f}, {pc_np[:, 2].max():.2f}]",
            "Center": f"[{pc_np[:, 0].mean():.2f}, {pc_np[:, 1].mean():.2f}, {pc_np[:, 2].mean():.2f}]",
        }

        # Add class distribution if change_map is provided
        if change_map is not None:
            unique_classes, class_counts = torch.unique(change_map, return_counts=True)

            # Convert to numpy for display
            unique_classes = unique_classes.cpu().numpy()
            class_counts = class_counts.cpu().numpy()

            # Calculate distribution
            total_points = change_map.numel()
            class_distribution = []

            for cls, count in zip(unique_classes, class_counts):
                percentage = (count / total_points) * 100
                cls_key = cls.item() if hasattr(cls, 'item') else cls

                if class_names and cls_key in class_names:
                    class_label = class_names[cls_key]
                    class_distribution.append({
                        "class_id": cls_key,
                        "class_name": class_label,
                        "count": int(count),
                        "percentage": percentage
                    })
                else:
                    class_distribution.append({
                        "class_id": cls_key,
                        "class_name": f"Class {cls_key}",
                        "count": int(count),
                        "percentage": percentage
                    })

            stats["class_distribution"] = class_distribution

        return stats
    except Exception as e:
        print(f"Error calculating point cloud stats: {e}")
        return {"error": str(e)}

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
