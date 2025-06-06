import torch
import numpy as np
import data.collators.buffer.cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import data.collators.buffer.cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors
from models.point_cloud_registration.buffer.point_learner import architecture


def batch_grid_subsampling_kpconv(points, batches_len, features=None, labels=None, sampleDl=0.1, max_p=0, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    """
    if (features is None) and (labels is None):
        device = points.device
        points = points.detach().cpu().numpy()
        batches_len = batches_len.detach().cpu().numpy().astype(np.int32)
        s_points, s_len = cpp_subsampling.subsample_batch(
            points, batches_len, sampleDl=sampleDl, max_p=max_p, verbose=verbose,
        )
        return (
            torch.from_numpy(s_points).to(device),
            torch.from_numpy(s_len).to(device),
        )

    elif (labels is None):
        device = points.device
        points = points.detach().cpu().numpy()
        batches_len = batches_len.detach().cpu().numpy().astype(np.int32)
        features = features.detach().cpu().numpy()
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(
            points, batches_len, features=features, sampleDl=sampleDl, max_p=max_p, verbose=verbose,
        )
        return (
            torch.from_numpy(s_points).to(device),
            torch.from_numpy(s_len).to(device),
            torch.from_numpy(s_features).to(device),
        )

    elif (features is None):
        device = points.device
        points = points.detach().cpu().numpy()
        batches_len = batches_len.detach().cpu().numpy().astype(np.int32)
        labels = labels.detach().cpu().numpy()
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(
            points, batches_len, classes=labels, sampleDl=sampleDl, max_p=max_p, verbose=verbose,
        )
        return (
            torch.from_numpy(s_points).to(device),
            torch.from_numpy(s_len).to(device),
            torch.from_numpy(s_labels).to(device),
        )

    else:
        device = points.device
        points = points.detach().cpu().numpy()
        batches_len = batches_len.detach().cpu().numpy().astype(np.int32)
        features = features.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(
            points, batches_len, features=features, classes=labels, sampleDl=sampleDl, max_p=max_p, verbose=verbose,
        )
        return (
            torch.from_numpy(s_points).to(device),
            torch.from_numpy(s_len).to(device),
            torch.from_numpy(s_features).to(device),
            torch.from_numpy(s_labels).to(device),
        )


def batch_neighbors_kpconv(queries, supports, q_batches, s_batches, radius, max_neighbors):
    """
    Computes neighbors for a batch of queries and supports, apply radius search
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B) the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """
    device = queries.device
    queries = queries.detach().cpu().numpy()
    supports = supports.detach().cpu().numpy()
    q_batches = q_batches.detach().cpu().numpy().astype(np.int32)
    s_batches = s_batches.detach().cpu().numpy().astype(np.int32)
    neighbors = cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)
    if max_neighbors > 0:
        return torch.from_numpy(neighbors[:, :max_neighbors]).to(device)
    else:
        return torch.from_numpy(neighbors).to(device)


def buffer_collate_fn(datapoint, config, neighborhood_limits):
    assert isinstance(datapoint, list)
    assert len(datapoint) == 1
    datapoint = datapoint[0]

    src_fds_points = datapoint['inputs']['src_pc_fds']['pos']
    tgt_fds_points = datapoint['inputs']['tgt_pc_fds']['pos']
    src_sds_points = datapoint['inputs']['src_pc_sds']['pos']
    tgt_sds_points = datapoint['inputs']['tgt_pc_sds']['pos']
    src_features = datapoint['inputs']['src_pc_sds']['normals']
    tgt_features = datapoint['inputs']['tgt_pc_sds']['normals']
    relt_pose = datapoint['labels']['transform']

    batched_points = torch.cat([src_sds_points, tgt_sds_points], dim=0)
    batched_lengths = torch.tensor([len(src_sds_points), len(tgt_sds_points)], dtype=torch.int64, device=batched_points.device)
    batched_features = torch.cat([src_features, tgt_features], dim=0)

    # Starting radius of convolutions
    r_normal = config.data.voxel_size_0 * config.point.conv_radius

    # Starting layer
    layer = 0

    # Lists of inputs
    input_points = []
    input_lengths = []
    input_neighbors = []
    input_downsamples = []
    input_upsamples = []

    for block_i, block in enumerate(architecture):
        # Stop when meeting a global pooling or upsampling
        if 'global' in block or 'upsample' in block:
            break

        # Check if the current block is a pooling or strided operation
        is_pooling_or_strided = 'pool' in block or 'strided' in block

        # If the current block is not a pooling or strided operation, add it to the layer blocks
        if not is_pooling_or_strided:
            # Check if the next block is not an upsampling operation
            if block_i < len(architecture) - 1 and not ('upsample' in architecture[block_i + 1]):
                continue

        # *****************************
        # Convolution neighbors indices
        # *****************************

        # Compute neighbor indices for convolution
        if not is_pooling_or_strided:
            neighbor_indices = batch_neighbors_kpconv(
                batched_points, batched_points, batched_lengths, batched_lengths,
                r_normal, neighborhood_limits[layer],
            )
        else:
            neighbor_indices = torch.zeros((0, 1), dtype=torch.int64)

        # *************************
        # Pooling neighbors indices
        # *************************

        # If the current block is a pooling operation, compute downsampling and upsampling indices
        if is_pooling_or_strided:
            downsample_points, downsample_lengths = batch_grid_subsampling_kpconv(
                batched_points, batched_lengths,
                sampleDl=2 * r_normal / config.point.conv_radius,
            )
            downsample_indices = batch_neighbors_kpconv(
                downsample_points, batched_points, downsample_lengths, batched_lengths,
                r_normal, neighborhood_limits[layer],
            )
            upsample_indices = batch_neighbors_kpconv(
                batched_points, downsample_points, batched_lengths, downsample_lengths,
                2 * r_normal, neighborhood_limits[layer],
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

        # Update radius and reset blocks
        r_normal *= 2.0
        layer += 1

    ###############
    # Return inputs
    ###############
    dict_inputs = {
        'src_pcd_raw': src_fds_points,
        'tgt_pcd_raw': tgt_fds_points,
        'src_pcd': src_sds_points,
        'tgt_pcd': tgt_sds_points,
        'features': batched_features.float(),
        'relt_pose': relt_pose,
        'points': input_points,
        'stack_lengths': input_lengths,
        'neighbors': input_neighbors,
        'pools': input_downsamples,
        'upsamples': input_upsamples,
    }

    return {
        'inputs': dict_inputs,
        'labels': datapoint['labels'],
        'meta_info': datapoint['meta_info'],
    }
