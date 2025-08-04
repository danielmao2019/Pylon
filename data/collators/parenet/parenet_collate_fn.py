from typing import Dict, Any, List
import torch
from data.collators.parenet.data import registration_collate_fn_stack_mode, precompute_neibors


def unpack_parenet_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Unpack Pylon data format to PARENet expected format."""
    # Extract points and features from nested structure
    src_pc = data['inputs']['src_pc']
    tgt_pc = data['inputs']['tgt_pc']
    
    # Handle different input formats (dict with 'pos' or direct tensor)
    if isinstance(src_pc, dict) and 'pos' in src_pc:
        src_points = src_pc['pos']
        src_features = src_pc.get('features', torch.ones(src_pc['pos'].shape[0], 1))
    else:
        src_points = src_pc
        src_features = torch.ones(src_pc.shape[0], 1)
        
    if isinstance(tgt_pc, dict) and 'pos' in tgt_pc:
        tgt_points = tgt_pc['pos']
        tgt_features = tgt_pc.get('features', torch.ones(tgt_pc['pos'].shape[0], 1))
    else:
        tgt_points = tgt_pc
        tgt_features = torch.ones(tgt_pc.shape[0], 1)
    
    # Convert to PARENet expected format (ref = target, src = source)
    parenet_dict = {
        'ref_points': tgt_points,  # PARENet uses ref for target
        'src_points': src_points,
        'ref_feats': tgt_features,
        'src_feats': src_features,
    }
    
    # Add transform if available
    if 'transform' in data['labels']:
        parenet_dict['transform'] = data['labels']['transform']
        
    # Add any meta info
    parenet_dict.update(data.get('meta_info', {}))
    
    return parenet_dict


def pack_parenet_results(parenet_collated: Dict[str, Any], original_data: Dict[str, Any]) -> Dict[str, Any]:
    """Pack PARENet collated results back to Pylon format."""
    
    # Map PARENet outputs to Pylon inputs structure
    inputs_dict = {}
    
    # Copy all PARENet-specific data to inputs for model access
    for key, value in parenet_collated.items():
        if key not in ['transform']:  # Keep transform for labels
            inputs_dict[key] = value
    
    # Prepare labels - transform should be in both inputs (for model) and labels (for evaluation)  
    labels_dict = {}
    if 'transform' in parenet_collated:
        labels_dict['transform'] = parenet_collated['transform']
        inputs_dict['transform'] = parenet_collated['transform']  # Model also needs this
    
    return {
        'inputs': inputs_dict,
        'labels': labels_dict,
        'meta_info': original_data.get('meta_info', {}),
    }


def parenet_collate_fn(
    data_dicts: List[Dict[str, Dict[str, Any]]],
    num_stages: int = 4,
    voxel_size: float = 0.05,
    subsample_ratio: float = 4.0,
    num_neighbors: List[int] = None,
    precompute_data: bool = True
) -> Dict[str, Dict[str, Any]]:
    """Collate function for PARENet in stack mode following Pylon patterns.
    
    Args:
        data_dicts: List of Pylon datapoints with 'inputs', 'labels', 'meta_info'
        num_stages: Number of hierarchical stages for multi-scale processing  
        voxel_size: Base voxel size for grid subsampling
        subsample_ratio: Ratio between consecutive stages
        num_neighbors: List of neighbor counts for each stage
        precompute_data: Whether to precompute multi-scale data and neighbors
        
    Returns:
        collated_dict: Pylon-formatted collated data with PARENet data in inputs
    """
    # Input validation
    assert isinstance(data_dicts, list), 'data_dicts must be a list'
    assert all(isinstance(data_dict, dict) for data_dict in data_dicts), \
        'data_dicts must be a list of dictionaries'
    assert all(data_dict.keys() >= {'inputs', 'labels', 'meta_info'} for data_dict in data_dicts), \
        'data_dicts must contain the keys inputs, labels, meta_info'
    assert isinstance(num_stages, int), 'num_stages must be an integer'
    assert isinstance(voxel_size, (int, float)), 'voxel_size must be a number'
    assert isinstance(subsample_ratio, (int, float)), 'subsample_ratio must be a number'
    assert isinstance(precompute_data, bool), 'precompute_data must be a boolean'
    
    # Initialize default neighbor counts if not provided
    if num_neighbors is None:
        num_neighbors = [32, 32, 32, 32][:num_stages]
    assert len(num_neighbors) == num_stages, \
        f"num_neighbors length ({len(num_neighbors)}) must match num_stages ({num_stages})"
    
    # Convert Pylon datapoints to PARENet format
    parenet_datapoints = []
    for datapoint in data_dicts:
        parenet_dict = unpack_parenet_data(datapoint)
        parenet_datapoints.append(parenet_dict)
    
    # Use original PARENet collation function
    parenet_collated = registration_collate_fn_stack_mode(
        parenet_datapoints,
        num_stages=num_stages,
        voxel_size=voxel_size,
        num_neighbors=num_neighbors,
        subsample_ratio=subsample_ratio,
        precompute_data=precompute_data
    )
    
    # Convert back to Pylon format
    return pack_parenet_results(parenet_collated, data_dicts[0])


def add_parenet_neighbors(batch_dict: Dict[str, Any], num_stages: int = 4, num_neighbors: List[int] = None) -> Dict[str, Any]:
    """Add neighbor computation to PARENet batch after CUDA transfer.
    
    This function should be called after the batch is moved to CUDA device.
    
    Args:
        batch_dict: Batch dictionary with 'inputs' containing PARENet data
        num_stages: Number of hierarchical stages
        num_neighbors: List of neighbor counts for each stage
        
    Returns:
        Updated batch dictionary with neighbor information added to inputs
    """
    if num_neighbors is None:
        num_neighbors = [32, 32, 32, 32][:num_stages]
    
    inputs = batch_dict['inputs']
    
    # Ensure we have the required keys from preprocessing
    if 'points' not in inputs or 'lengths' not in inputs:
        raise ValueError("Batch must contain 'points' and 'lengths' from preprocessing")
    
    # Compute neighbors on GPU tensors
    neighbors_dict = precompute_neibors(
        inputs['points'], 
        inputs['lengths'], 
        num_stages, 
        num_neighbors
    )
    
    # Add neighbor information to inputs
    inputs.update(neighbors_dict)
    
    return batch_dict