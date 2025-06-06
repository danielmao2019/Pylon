from typing import Dict, Any
import torch
from data.collators.geotransformer.grid_subsample import grid_subsample
from data.collators.geotransformer.radius_search import radius_search
from data.collators.pcr_collator import pcr_collate_fn


def geotransformer_collate_fn(
    data_dicts, num_stages, voxel_size, search_radius, neighbor_limits, precompute_data=True
) -> Dict[str, Dict[str, Any]]:
    r"""Collate function for registration in stack mode.

    Args:
        data_dicts (List[Dict[str, Dict[str, Any]]]): List of datapoints, where each datapoint is a dict with:
            - inputs: Dict containing 'src_pc', 'tgt_pc', and 'correspondences'
            - labels: Dict containing 'transform'
            - meta_info: Dict containing metadata
        num_stages (int): Number of stages for multi-scale processing
        voxel_size (float): Initial voxel size for grid subsampling
        search_radius (float): Initial search radius for neighbor search
        neighbor_limits (List[int]): List of neighbor limits for each stage
        precompute_data (bool): Whether to precompute multi-scale data

    Returns:
        collated_dict (Dict): Collated data dictionary
    """
    # Input checks
    assert isinstance(data_dicts, list), 'data_dicts must be a list'
    assert all(isinstance(data_dict, dict) for data_dict in data_dicts), \
        'data_dicts must be a list of dictionaries'
    assert all(data_dict.keys() >= {'inputs', 'labels', 'meta_info'} for data_dict in data_dicts), \
        'data_dicts must contain the keys inputs, labels, meta_info'
    assert isinstance(num_stages, int), 'num_stages must be an integer'
    assert isinstance(voxel_size, (int, float)), 'voxel_size must be a float'
    assert isinstance(search_radius, (int, float)), 'search_radius must be a float'
    assert isinstance(neighbor_limits, list), f'neighbor_limits must be a list. Got {type(neighbor_limits)}.'
    assert all(isinstance(limit, int) for limit in neighbor_limits), 'neighbor_limits must be a list of integers'
    assert isinstance(precompute_data, bool), 'precompute_data must be a boolean'

    # Main logic
    batch_size = len(data_dicts)
    assert batch_size == 1
    data = data_dicts[0]  # Get the single item directly
    device = data['inputs']['src_pc']['pos'].device

    # Prepare batched data
    feats = torch.cat([data['inputs']['tgt_pc']['feat'], data['inputs']['src_pc']['feat']], dim=0)
    points = torch.cat([data['inputs']['tgt_pc']['pos'], data['inputs']['src_pc']['pos']], dim=0)
    lengths = torch.tensor([len(data['inputs']['tgt_pc']['pos']), len(data['inputs']['src_pc']['pos'])], 
                         dtype=torch.long, device=device)

    # Define architecture for pcr_collator
    architecture = []
    current_voxel_size = voxel_size
    current_radius = search_radius
    
    for i in range(num_stages):
        # Add conv block
        architecture.append({
            'type': 'conv',
            'radius': current_radius,
            'sample_dl': current_voxel_size,
            'neighborhood_limit': neighbor_limits[i]
        })
        
        # Add pool block if not last stage
        if i < num_stages - 1:
            architecture.append({
                'type': 'pool',
                'radius': current_radius,
                'sample_dl': current_voxel_size * 2,  # Double the voxel size for pooling
                'neighborhood_limit': neighbor_limits[i]
            })
        
        current_voxel_size *= 2
        current_radius *= 2

    # Call pcr_collator
    collated_data = pcr_collate_fn(
        points, points,  # Use same points for src and tgt since we're doing self-neighborhood
        architecture,
        downsample_fn=grid_subsample,
        neighbor_fn=radius_search,
    )

    # Remove last elements from downsamples and upsamples
    collated_data['downsamples'] = collated_data['downsamples'][:-1]
    collated_data['upsamples'] = collated_data['upsamples'][:-1]

    # Map keys to match original format
    inputs_dict = {
        'points': collated_data['points'],
        'lengths': collated_data['lengths'],
        'neighbors': collated_data['neighbors'],
        'subsampling': collated_data['downsamples'],  # Map downsamples to subsampling
        'upsampling': collated_data['upsamples'],  # Map upsamples to upsampling
        'features': feats,
        'transform': data['labels']['transform']
    }

    # Prepare meta info
    meta_info = {
        key: [data['meta_info'][key]]
        for key in data['meta_info']
    }
    meta_info['batch_size'] = batch_size

    return {
        'inputs': inputs_dict,
        'labels': {
            'transform': data['labels']['transform'].unsqueeze(0),  # Add batch dimension
        },
        'meta_info': meta_info,
    }
