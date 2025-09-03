import os
from pathlib import Path
import torch
import json
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.pipelines.base_pipeline import Pipeline
from utils.gs_ops.apply_transform import apply_transform_to_gs


def load_gs(model_dir: str, device: torch.device = None) -> Pipeline:
    """Load a Gaussian Splatting model from a model directory.
    
    Args:
        model_dir: Path to the model directory containing config.yml and dataparser_transforms.json
        device: Target device for the model. If None, uses the pipeline's default device.
        
    Returns:
        Loaded Pipeline object containing the Gaussian Splatting model
    """
    # Construct file paths
    gs_filepath = os.path.join(model_dir, "config.yml")
    transforms_filepath = os.path.join(model_dir, "dataparser_transforms.json")
    
    # Load the pipeline
    _, pipeline, _, _ = eval_setup(Path(gs_filepath), test_mode="test")
    
    # Use specified device or fallback to pipeline's device
    if device is None:
        device = pipeline.model.device
    else:
        # Move the model to the specified device
        pipeline.model = pipeline.model.to(device)

    # Load transforms if available
    if os.path.isfile(transforms_filepath):
        with open(transforms_filepath, 'r') as f:
            transforms = json.load(f)
        rotation, translation = torch.split(torch.tensor(transforms['transform'], device=device), [3, 1], dim=-1)
        scale = transforms['scale']
        # Invert transform
        rotation = rotation.T
        translation = -translation
        scale = 1.0 / scale
    else:
        rotation, translation, scale = None, None, None

    # Apply transforms to the model
    with torch.no_grad():
        apply_transform_to_gs(pipeline, rotation=rotation, translation=translation, scale=scale)

        opencv_to_standard = torch.tensor([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0],
        ], dtype=torch.float32, device=device)
        apply_transform_to_gs(pipeline, rotation=opencv_to_standard, translation=None, scale=None)

    return pipeline
