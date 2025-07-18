from typing import List, Dict, Callable, Any
import torch


def pcr_collate_fn(
    src_points: torch.Tensor,
    tgt_points: torch.Tensor,
    architecture: List[Dict[str, Any]],
    downsample_fn: Callable, 
    neighbor_fn: Callable
) -> Dict[str, List[torch.Tensor]]:
    assert isinstance(src_points, torch.Tensor)
    assert isinstance(tgt_points, torch.Tensor)
    assert isinstance(architecture, list)
    assert all(isinstance(block, dict) for block in architecture)
    assert all(block.keys() == {
        'neighbor', 'neighbor_radius', 'neighbor_neighborhood_limit',
        'downsample', 'sample_dl',
        'downsample_radius', 'downsample_neighborhood_limit',
        'upsample_radius', 'upsample_neighborhood_limit',
    } for block in architecture)
    assert isinstance(downsample_fn, Callable)
    assert isinstance(neighbor_fn, Callable)

    batched_points = torch.cat([src_points, tgt_points], dim=0)
    batched_lengths = torch.tensor([len(src_points), len(tgt_points)], dtype=torch.int64, device=batched_points.device)

    # Lists of inputs
    input_points = []
    input_lengths = []
    input_neighbors = []
    input_downsamples = []
    input_upsamples = []

    for block in architecture:
        # *****************************
        # Convolution neighbors indices
        # *****************************

        # Compute neighbor indices for convolution
        if block['neighbor']:
            neighbor_indices = neighbor_fn(
                batched_points, batched_points, batched_lengths, batched_lengths,
                block['neighbor_radius'], block['neighbor_neighborhood_limit'],
            )
        else:
            neighbor_indices = torch.zeros((0, 1), dtype=torch.int64)

        # *************************
        # Pooling neighbors indices
        # *************************

        # If the current block is a pooling operation, compute downsampling and upsampling indices
        if block['downsample']:
            downsample_points, downsample_lengths = downsample_fn(
                batched_points, batched_lengths,
                block['sample_dl'],
            )
            downsample_indices = neighbor_fn(
                downsample_points, batched_points, downsample_lengths, batched_lengths,
                block['downsample_radius'], block['downsample_neighborhood_limit'],
            )
            upsample_indices = neighbor_fn(
                batched_points, downsample_points, batched_lengths, downsample_lengths,
                block['upsample_radius'], block['upsample_neighborhood_limit'],
            )
        else:
            downsample_points = torch.zeros((0, 3), dtype=torch.float32)
            downsample_lengths = torch.zeros((0,), dtype=torch.int64)
            downsample_indices = torch.zeros((0, 1), dtype=torch.int64)
            upsample_indices = torch.zeros((0, 1), dtype=torch.int64)

        # Updating input lists
        input_points += [batched_points.float()]
        input_lengths += [batched_lengths]
        input_neighbors += [neighbor_indices.long()]
        input_downsamples += [downsample_indices.long()]
        input_upsamples += [upsample_indices.long()]

        # New points for next layer
        batched_points = downsample_points
        batched_lengths = downsample_lengths

    ###############
    # Return inputs
    ###############
    return {
        'points': input_points,
        'lengths': input_lengths,
        'neighbors': input_neighbors,
        'downsamples': input_downsamples,
        'upsamples': input_upsamples,
    }
