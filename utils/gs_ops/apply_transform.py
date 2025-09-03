"""Apply transformations to Gaussian Splatting models."""

import math
import torch
from typing import Optional
from nerfstudio.pipelines.base_pipeline import Pipeline


def apply_transform_to_gs(
    pipeline: Pipeline, 
    rotation: Optional[torch.Tensor] = None, 
    translation: Optional[torch.Tensor] = None, 
    scale: Optional[float] = None
) -> None:
    """Apply scale, translation, and rotation transformations to a Gaussian Splatting model.
    
    Transformations are applied in order: scale, translation, rotation.
    This modifies the pipeline's model in-place.
    
    Args:
        pipeline: Nerfstudio Pipeline containing the Gaussian Splatting model
        rotation: Optional 3x3 rotation matrix as torch.Tensor
        translation: Optional 3D translation vector as torch.Tensor
        scale: Optional scale factor as float
    """
    assert isinstance(pipeline, Pipeline), f"pipeline must be Pipeline, got {type(pipeline)}"
    
    with torch.no_grad():
        # Apply scale first
        if scale is not None:
            assert isinstance(scale, float), f"scale must be float, got {type(scale)}"
            pipeline.model.means[:] *= scale
            pipeline.model.scales[:] += math.log(scale)
            pipeline.model.opacities[:] *= 1.0 / (scale ** 3)
        
        # Apply translation second
        if translation is not None:
            assert isinstance(translation, torch.Tensor), f"translation must be torch.Tensor, got {type(translation)}"
            assert translation.numel() == 3, f"translation must have 3 elements, got {translation.numel()}"
            translation = translation.view(1, 3)
            pipeline.model.means[:] += translation
        
        # Apply rotation last
        if rotation is not None:
            assert isinstance(rotation, torch.Tensor), f"rotation must be torch.Tensor, got {type(rotation)}"
            assert rotation.shape == (3, 3), f"rotation must be 3x3 matrix, got shape {rotation.shape}"
            
            # Rotate means
            pipeline.model.means[:] = (rotation @ pipeline.model.means.T).T
            
            # Rotate scales (covariances)
            actual_scales = torch.exp(pipeline.model.scales)
            covariances = torch.diag_embed(actual_scales)
            transformed_covariances = rotation @ covariances @ rotation.T
            pipeline.model.scales[:] = torch.log(torch.diagonal(transformed_covariances, dim1=-2, dim2=-1))
