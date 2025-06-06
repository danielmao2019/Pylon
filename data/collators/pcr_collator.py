from typing import List, Dict, Callable, Any
import torch


def pcr_collate_fn(src_points, tgt_points, architecture: List[Dict[str, Any]], downsample_fn: Callable, neighbor_fn: Callable):
    assert isinstance(src_points, torch.Tensor)
    assert isinstance(tgt_points, torch.Tensor)
    assert src_points.shape[0] == tgt_points.shape[0]
    assert isinstance(architecture, list)
    assert all(isinstance(block, dict) for block in architecture)
    assert all(block.keys() == {'type', 'radius', 'sample_dl', 'neighborhood_limit'} for block in architecture)
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

    for block_i, block in enumerate(architecture):
        # Stop when meeting a global pooling or upsampling
        if 'global' in block['type'] or 'upsample' in block['type']:
            break

        # Check if the current block is a pooling or strided operation
        is_pooling_or_strided = 'pool' in block['type'] or 'strided' in block['type']

        # If the current block is not a pooling or strided operation, add it to the layer blocks
        if not is_pooling_or_strided:
            # Check if the next block is not an upsampling operation
            if block_i < len(architecture) - 1 and not ('upsample' in architecture[block_i + 1]['type']):
                continue

        # *****************************
        # Convolution neighbors indices
        # *****************************

        # Compute neighbor indices for convolution
        if not is_pooling_or_strided:
            neighbor_indices = neighbor_fn(
                batched_points, batched_points, batched_lengths, batched_lengths,
                block['radius'], block['neighborhood_limit'],
            )
        else:
            neighbor_indices = torch.zeros((0, 1), dtype=torch.int64)

        # *************************
        # Pooling neighbors indices
        # *************************

        # If the current block is a pooling operation, compute downsampling and upsampling indices
        if is_pooling_or_strided:
            downsample_points, downsample_lengths = downsample_fn(
                batched_points, batched_lengths,
                sampleDl=block['sample_dl'],
            )
            downsample_indices = neighbor_fn(
                downsample_points, batched_points, downsample_lengths, batched_lengths,
                block['radius'], block['neighborhood_limit'],
            )
            upsample_indices = neighbor_fn(
                batched_points, downsample_points, batched_lengths, downsample_lengths,
                2 * block['radius'], block['neighborhood_limit'],
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
