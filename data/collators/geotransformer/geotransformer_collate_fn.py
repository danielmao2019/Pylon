from typing import Dict, Any
import torch
from data.collators.geotransformer.grid_subsample import grid_subsample
from data.collators.geotransformer.radius_search import radius_search


def precompute_data_stack_mode(points, lengths, num_stages, voxel_size, radius, neighbor_limits):
    assert num_stages == len(neighbor_limits)

    points_list = []
    lengths_list = []
    neighbors_list = []
    subsampling_list = []
    upsampling_list = []

    # grid subsampling
    for i in range(num_stages):
        if i > 0:
            points, lengths = grid_subsample(points, lengths, voxel_size=voxel_size)
        points_list.append(points)
        lengths_list.append(lengths)
        voxel_size *= 2

    # radius search
    for i in range(num_stages):
        cur_points = points_list[i]
        cur_lengths = lengths_list[i]

        neighbors = radius_search(
            cur_points,
            cur_points,
            cur_lengths,
            cur_lengths,
            radius,
            neighbor_limits[i],
        )
        # Validate neighbor tensor shape
        expected_shape = (cur_points.shape[0], neighbor_limits[i])
        assert neighbors.shape == expected_shape, \
            f"Stage {i}: Expected shape {expected_shape} but got {neighbors.shape}"
        neighbors_list.append(neighbors)

        if i < num_stages - 1:
            sub_points = points_list[i + 1]
            sub_lengths = lengths_list[i + 1]

            subsampling = radius_search(
                sub_points,
                cur_points,
                sub_lengths,
                cur_lengths,
                radius,
                neighbor_limits[i],
            )
            # Validate subsampling tensor shape
            expected_shape = (sub_points.shape[0], neighbor_limits[i])
            assert subsampling.shape == expected_shape, \
                f"Stage {i} subsampling: Expected shape {expected_shape} but got {subsampling.shape}"
            subsampling_list.append(subsampling)

            upsampling = radius_search(
                cur_points,
                sub_points,
                cur_lengths,
                sub_lengths,
                radius * 2,
                neighbor_limits[i + 1],
            )
            # Validate upsampling tensor shape
            expected_shape = (cur_points.shape[0], neighbor_limits[i + 1])
            assert upsampling.shape == expected_shape, \
                f"Stage {i} upsampling: Expected shape {expected_shape} but got {upsampling.shape}"
            upsampling_list.append(upsampling)

        radius *= 2

    return {
        'points': points_list,
        'lengths': lengths_list,
        'neighbors': neighbors_list,
        'subsampling': subsampling_list,
        'upsampling': upsampling_list,
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
    assert isinstance(search_radius, (int, float)), 'search_radius must be a float'
    assert isinstance(neighbor_limits, list), f'neighbor_limits must be a list. Got {type(neighbor_limits)}.'
    assert all(isinstance(limit, int) for limit in neighbor_limits), 'neighbor_limits must be a list of integers'
    assert isinstance(precompute_data, bool), 'precompute_data must be a boolean'

    # Main logic
    batch_size = len(data_dicts)
    assert batch_size == 1
    device = data_dicts[0]['inputs']['src_pc']['pos'].device

    feats = torch.cat([dd['inputs']['tgt_pc']['feat'] for dd in data_dicts] + [dd['inputs']['src_pc']['feat'] for dd in data_dicts], dim=0)
    points_list = [dd['inputs']['tgt_pc']['pos'] for dd in data_dicts] + [dd['inputs']['src_pc']['pos'] for dd in data_dicts]
    lengths = torch.tensor([points.shape[0] for points in points_list], dtype=torch.long, device=device)
    points = torch.cat(points_list, dim=0)

    # Process source and target point clouds separately
    inputs_dict = precompute_data_stack_mode(points, lengths, num_stages, voxel_size, search_radius, neighbor_limits)
    inputs_dict['features'] = feats
    inputs_dict['transform'] = data_dicts[0]['labels']['transform']
    collated_dict = {
        'inputs': inputs_dict,
        'labels': {
            'transform': torch.stack([d['labels']['transform'] for d in data_dicts], dim=0),
        },
        'meta_info': {
            'idx': torch.tensor([d['meta_info']['idx'] for d in data_dicts], device=device),
            'point_indices': [d['meta_info']['point_indices'] for d in data_dicts],
            'filepath': [d['meta_info']['filepath'] for d in data_dicts],
            'batch_size': batch_size,
        },
    }
    return collated_dict
