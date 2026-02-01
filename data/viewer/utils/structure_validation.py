"""Structure validation utilities for dataset display patterns.

This module provides validation functions to check if datapoint structures match
the expected format for each dataset group before calling predefined display functions.
"""
from typing import Any, Dict
import torch
from data.structures.three_d.point_cloud.point_cloud import PointCloud


def _validate_basic_datapoint_structure(datapoint: Dict[str, Any]) -> None:
    """Validate basic datapoint structure common to all dataset types.

    Args:
        datapoint: Datapoint dictionary to validate

    Raises:
        AssertionError: If validation fails
    """
    # Input validations
    assert isinstance(datapoint, dict), f"{type(datapoint)=}"

    # Check required top-level keys
    required_keys = {'inputs', 'labels', 'meta_info'}
    missing_keys = required_keys - set(datapoint.keys())
    assert not missing_keys, f"Missing required top-level keys: {missing_keys}"

    # Validate inputs structure
    assert isinstance(datapoint['inputs'], dict), f"'inputs' must be a dictionary, got {type(datapoint['inputs'])}"
    assert datapoint['inputs'], "'inputs' dictionary must not be empty"

    # Validate labels structure
    assert isinstance(datapoint['labels'], dict), f"'labels' must be a dictionary, got {type(datapoint['labels'])}"
    assert datapoint['labels'], "'labels' dictionary must not be empty"

    # Validate meta_info structure
    assert isinstance(datapoint['meta_info'], dict), f"'meta_info' must be a dictionary, got {type(datapoint['meta_info'])}"


def validate_semseg_structure(datapoint: Dict[str, Any]) -> None:
    """Validate semantic segmentation datapoint structure.

    Expected structure:
    - inputs: {'image': torch.Tensor}  # Shape: [C, H, W] or [H, W, C]
    - labels: {'label': torch.Tensor or Dict}  # Tensor: [H, W], Dict: instance segmentation format

    Args:
        datapoint: Datapoint dictionary to validate

    Raises:
        AssertionError: If validation fails
    """
    # Input validations
    assert isinstance(datapoint, dict), f"{type(datapoint)=}"

    # Basic structure validation
    _validate_basic_datapoint_structure(datapoint)

    # Validate inputs - check for required image field
    inputs = datapoint['inputs']
    assert 'image' in inputs, f"inputs must have 'image' key, got keys: {list(inputs.keys())}"

    # Validate image tensor
    image = inputs['image']
    assert isinstance(image, torch.Tensor), f"inputs['image'] must be torch.Tensor, got {type(image)}"
    assert image.ndim == 3, f"inputs['image'] must be 3D tensor [C,H,W], got {image.ndim}D with shape {image.shape}"
    assert image.numel() > 0, "inputs['image'] must not be empty tensor"

    # Validate labels - check for required label field
    labels = datapoint['labels']
    assert 'label' in labels, f"labels must have 'label' key, got keys: {list(labels.keys())}"

    # Validate label (allow both tensor and dict formats for instance segmentation)
    label = labels['label']
    if isinstance(label, torch.Tensor):
        assert label.ndim in [2, 3], f"labels['label'] tensor must be 2D [H,W] or 3D [C,H,W], got {label.ndim}D with shape {label.shape}"
        assert label.numel() > 0, "labels['label'] must not be empty tensor"
    elif isinstance(label, dict):
        # Instance segmentation format validation
        assert 'masks' in label and 'indices' in label, f"labels['label'] dict must have 'masks' and 'indices' keys, got keys: {list(label.keys())}"
    else:
        assert False, f"labels['label'] must be torch.Tensor or dict, got {type(label)}"


def validate_2dcd_structure(datapoint: Dict[str, Any]) -> None:
    """Validate 2D change detection datapoint structure.

    Expected structure:
    - inputs: {'img_1': torch.Tensor, 'img_2': torch.Tensor}  # Shape: [C, H, W] or [H, W, C]
    - labels: {'change_map': torch.Tensor}  # Shape: [H, W] or [1, H, W]

    Args:
        datapoint: Datapoint dictionary to validate

    Raises:
        AssertionError: If validation fails
    """
    # Input validations
    assert isinstance(datapoint, dict), f"{type(datapoint)=}"

    # Basic structure validation
    _validate_basic_datapoint_structure(datapoint)

    # Validate inputs - check for required image fields
    inputs = datapoint['inputs']
    for img_key in ['img_1', 'img_2']:
        assert img_key in inputs, f"inputs must have '{img_key}' key, got keys: {list(inputs.keys())}"

        # Validate image tensor
        img = inputs[img_key]
        assert isinstance(img, torch.Tensor), f"inputs['{img_key}'] must be torch.Tensor, got {type(img)}"
        assert img.ndim == 3, f"inputs['{img_key}'] must be 3D tensor [C,H,W], got {img.ndim}D with shape {img.shape}"
        assert img.numel() > 0, f"inputs['{img_key}'] must not be empty tensor"

    # Check for shape consistency between images
    img1, img2 = inputs['img_1'], inputs['img_2']
    assert img1.shape == img2.shape, f"Image shapes don't match: img_1={img1.shape}, img_2={img2.shape}"

    # Validate labels - check for required change map field
    labels = datapoint['labels']
    assert 'change_map' in labels, f"labels must have 'change_map' key, got keys: {list(labels.keys())}"

    # Validate change map tensor
    change_map = labels['change_map']
    assert isinstance(change_map, torch.Tensor), f"labels['change_map'] must be torch.Tensor, got {type(change_map)}"
    assert change_map.ndim >= 2, f"labels['change_map'] must be at least 2D [H,W], got {change_map.ndim}D with shape {change_map.shape}"
    assert change_map.numel() > 0, "labels['change_map'] must not be empty tensor"


def validate_3dcd_structure(datapoint: Dict[str, Any]) -> None:
    """Validate 3D change detection datapoint structure.

    Expected structure:
    - inputs: {'pc_1': Dict, 'pc_2': Dict, [optional] 'kdtree_1': Any, 'kdtree_2': Any}
    - labels: {'change_map': torch.Tensor}  # Shape: [N] where N is number of points

    Args:
        datapoint: Datapoint dictionary to validate

    Raises:
        AssertionError: If validation fails
    """
    # Input validations
    assert isinstance(datapoint, dict), f"{type(datapoint)=}"

    # Basic structure validation
    _validate_basic_datapoint_structure(datapoint)

    # Validate inputs - check for required point cloud fields
    inputs = datapoint['inputs']
    pc_positions: Dict[str, torch.Tensor] = {}
    for pc_key in ['pc_1', 'pc_2']:
        assert pc_key in inputs, f"inputs must have '{pc_key}' key, got keys: {list(inputs.keys())}"

        pc = inputs[pc_key]
        assert isinstance(pc, PointCloud), f"inputs['{pc_key}'] must be PointCloud, got {type(pc)}"
        pos = pc.xyz
        pc_positions[pc_key] = pos

    # Check for shape consistency between point clouds
    pos1 = pc_positions['pc_1']
    pos2 = pc_positions['pc_2']
    assert pos1.shape[1] == pos2.shape[1], f"Point cloud dimension mismatch: pc_1 has {pos1.shape[1]}D points, pc_2 has {pos2.shape[1]}D points"

    # Validate labels - check for required change map field
    labels = datapoint['labels']
    assert 'change_map' in labels, f"labels must have 'change_map' key, got keys: {list(labels.keys())}"

    # Validate change map tensor
    change_map = labels['change_map']
    assert isinstance(change_map, torch.Tensor), f"labels['change_map'] must be torch.Tensor, got {type(change_map)}"
    assert change_map.ndim == 1, f"labels['change_map'] must be 1D tensor [N], got {change_map.ndim}D with shape {change_map.shape}"
    assert change_map.numel() > 0, "labels['change_map'] must not be empty tensor"


def validate_pcr_structure(datapoint: Dict[str, Any]) -> None:
    """Validate point cloud registration datapoint structure.

    Expected structure:
    - inputs: {'src_pc': Dict, 'tgt_pc': Dict, [optional] 'correspondences': torch.Tensor}
      OR: {'points': List[torch.Tensor], 'lengths'/'stack_lengths': List[torch.Tensor]} (batched format)
    - labels: {'transform': torch.Tensor}  # Shape: [4, 4] transformation matrix

    Args:
        datapoint: Datapoint dictionary to validate

    Raises:
        AssertionError: If validation fails
    """
    # Input validations
    assert isinstance(datapoint, dict), f"{type(datapoint)=}"

    # Basic structure validation
    _validate_basic_datapoint_structure(datapoint)

    # Validate inputs - check for batched vs single format
    inputs = datapoint['inputs']

    # Check if this is batched format (from collators)
    if 'points' in inputs and ('lengths' in inputs or 'stack_lengths' in inputs):
        # Batched format validation
        points = inputs['points']
        assert isinstance(points, torch.Tensor), f"inputs['points'] must be torch.Tensor, got {type(points)}"
        assert points.numel() > 0, "inputs['points'] must not be empty tensor"

        # Check lengths field
        length_key = 'lengths' if 'lengths' in inputs else 'stack_lengths'
        lengths = inputs[length_key]
        assert isinstance(lengths, torch.Tensor), f"inputs['{length_key}'] must be torch.Tensor, got {type(lengths)}"
        assert lengths.numel() > 0, f"inputs['{length_key}'] must not be empty tensor"
    else:
        # Single format validation - check for required point cloud fields
        pc_positions: Dict[str, torch.Tensor] = {}
        for pc_key in ['src_pc', 'tgt_pc']:
            assert pc_key in inputs, f"inputs must have '{pc_key}' key, got keys: {list(inputs.keys())}"

            pc = inputs[pc_key]
            assert isinstance(pc, PointCloud), f"inputs['{pc_key}'] must be PointCloud, got {type(pc)}"

            pos = pc.xyz
            pc_positions[pc_key] = pos

        # Check for optional correspondences field
        if 'correspondences' in inputs:
            corr = inputs['correspondences']
            assert isinstance(corr, torch.Tensor), f"inputs['correspondences'] must be torch.Tensor, got {type(corr)}"
            assert corr.ndim == 2, f"inputs['correspondences'] must be 2D tensor, got {corr.ndim}D with shape {corr.shape}"
            assert corr.numel() > 0, "inputs['correspondences'] must not be empty tensor"

    # Validate labels - check for required transform field
    labels = datapoint['labels']
    assert 'transform' in labels, f"labels must have 'transform' key, got keys: {list(labels.keys())}"

    # Validate transform tensor
    transform = labels['transform']
    assert isinstance(transform, torch.Tensor), f"labels['transform'] must be torch.Tensor, got {type(transform)}"
    assert transform.ndim == 2, f"labels['transform'] must be 2D tensor, got {transform.ndim}D with shape {transform.shape}"
    assert transform.shape[-2:] == (4, 4), f"labels['transform'] must have shape [4,4], got {transform.shape}"
    assert transform.numel() > 0, "labels['transform'] must not be empty tensor"


def validate_structure_for_type(dataset_type: str, datapoint: Dict[str, Any]) -> None:
    """Validate datapoint structure for a specific dataset type.

    Args:
        dataset_type: Type of dataset ('semseg', '2dcd', '3dcd', 'pcr')
        datapoint: Datapoint dictionary to validate

    Raises:
        ValueError: If dataset_type is not supported
        AssertionError: If validation fails
    """
    # Input validations
    assert isinstance(dataset_type, str), f"{type(dataset_type)=}"
    assert isinstance(datapoint, dict), f"{type(datapoint)=}"

    validators = {
        'semseg': validate_semseg_structure,
        '2dcd': validate_2dcd_structure,
        '3dcd': validate_3dcd_structure,
        'pcr': validate_pcr_structure,
    }

    assert dataset_type in validators, f"Unsupported dataset type for validation: {dataset_type}. Supported types: {list(validators.keys())}"

    validators[dataset_type](datapoint)
