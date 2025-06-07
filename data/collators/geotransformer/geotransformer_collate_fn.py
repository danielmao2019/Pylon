from typing import Dict, Any
import torch
from data.collators.geotransformer.grid_subsample import grid_subsample
from data.collators.geotransformer.radius_search import radius_search
from data.collators.pcr_collator import pcr_collate_fn


def unpack_geotransformer_data(data):
    """Unpack data to get points and features."""
    device = data['inputs']['src_pc']['pos'].device

    # Prepare batched data
    src_points = data['inputs']['src_pc']['pos']
    tgt_points = data['inputs']['tgt_pc']['pos']
    feats = torch.cat([data['inputs']['tgt_pc']['feat'], data['inputs']['src_pc']['feat']], dim=0)
    lengths = torch.tensor([len(src_points), len(tgt_points)], dtype=torch.long, device=device)

    return {
        'src_points': src_points,
        'tgt_points': tgt_points,
        'features': feats,
        'lengths': lengths,
        'transform': data['labels']['transform'],
        'meta_info': data['meta_info'],
    }


def create_geotransformer_architecture(num_stages, voxel_size, search_radius, neighbor_limits):
    """Create architecture for pcr_collator."""
    architecture = []
    current_voxel_size = voxel_size
    current_radius = search_radius
    
    for i in range(num_stages):
        architecture.append({
            'neighbor': True,
            'downsample': True,
            'radius': current_radius,
            'sample_dl': current_voxel_size,
            'neighborhood_limit': neighbor_limits[i]
        })
        current_voxel_size *= 2
        current_radius *= 2

    return architecture


def pack_geotransformer_results(collated_data, unpacked_data):
    """Pack pcr_collator results into geotransformer format."""
    # Remove last elements from downsamples and upsamples
    collated_data['downsamples'] = collated_data['downsamples'][:-1]
    collated_data['upsamples'] = collated_data['upsamples'][:-1]

    # Map keys to match original format
    return {
        'points': collated_data['points'],
        'lengths': collated_data['lengths'],
        'neighbors': collated_data['neighbors'],
        'subsampling': collated_data['downsamples'],  # Map downsamples to subsampling
        'upsampling': collated_data['upsamples'],  # Map upsamples to upsampling
        'features': unpacked_data['features'],
        'transform': unpacked_data['transform']
    }


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
    voxel_size *= 2  # keep consistent with the original implementation
    assert isinstance(search_radius, (int, float)), 'search_radius must be a float'
    assert isinstance(neighbor_limits, list), f'neighbor_limits must be a list. Got {type(neighbor_limits)}.'
    assert all(isinstance(limit, int) for limit in neighbor_limits), 'neighbor_limits must be a list of integers'
    assert isinstance(precompute_data, bool), 'precompute_data must be a boolean'

    # Main logic
    batch_size = len(data_dicts)
    assert batch_size == 1
    data = data_dicts[0]  # Get the single item directly

    # Unpack data
    unpacked_data = unpack_geotransformer_data(data)

    # Create architecture
    architecture = create_geotransformer_architecture(num_stages, voxel_size, search_radius, neighbor_limits)

    # Call pcr_collator
    collated_data = pcr_collate_fn(
        src_points=unpacked_data['tgt_points'],
        tgt_points=unpacked_data['src_points'],
        architecture=architecture,
        downsample_fn=grid_subsample,
        neighbor_fn=radius_search,
    )

    # Pack results
    inputs_dict = pack_geotransformer_results(collated_data, unpacked_data)

    # Prepare meta info
    meta_info = {
        key: [unpacked_data['meta_info'][key]]
        for key in unpacked_data['meta_info']
    }
    meta_info['batch_size'] = batch_size

    return {
        'inputs': inputs_dict,
        'labels': {
            'transform': unpacked_data['transform'].unsqueeze(0),  # Add batch dimension
        },
        'meta_info': meta_info,
    }
