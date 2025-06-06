import numpy as np
import torch
from data.collators.pcr_collator import pcr_collate_fn


def batch_grid_subsampling_kpconv(points, batches_len, features=None, labels=None, sampleDl=0.1, max_p=0, verbose=0, random_grid_orient=True):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    """
    device = points.device

    if (features is None) and (labels is None):
        s_points, s_len = cpp_subsampling.subsample_batch(points.cpu(),
                                                          batches_len.cpu(),
                                                          sampleDl=sampleDl,
                                                          max_p=max_p,
                                                          verbose=verbose)
        return (
            torch.from_numpy(s_points).to(device),
            torch.from_numpy(s_len).to(device)
        )

    elif (labels is None):
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(points.cpu(),
                                                                      batches_len.cpu(),
                                                                      features=features,
                                                                      sampleDl=sampleDl,
                                                                      max_p=max_p,
                                                                      verbose=verbose)
        return (
            torch.from_numpy(s_points).to(device),
            torch.from_numpy(s_len).to(device),
            torch.from_numpy(s_features).to(device)
        )

    elif (features is None):
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(points.cpu(),
                                                                    batches_len.cpu(),
                                                                    classes=labels,
                                                                    sampleDl=sampleDl,
                                                                    max_p=max_p,
                                                                    verbose=verbose)
        return (
            torch.from_numpy(s_points).to(device),
            torch.from_numpy(s_len).to(device),
            torch.from_numpy(s_labels).to(device)
        )

    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(points.cpu(),
                                                                              batches_len.cpu(),
                                                                              features=features,
                                                                              classes=labels,
                                                                              sampleDl=sampleDl,
                                                                              max_p=max_p,
                                                                              verbose=verbose)
        return (
            torch.from_numpy(s_points).to(device),
            torch.from_numpy(s_len).to(device),
            torch.from_numpy(s_features).to(device),
            torch.from_numpy(s_labels).to(device)
        )

def batch_neighbors_kpconv(queries, supports, q_batches, s_batches, radius, max_neighbors):
    """
    Computes neighbors for a batch of queries and supports, apply radius search
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """
    device = queries.device

    neighbors = cpp_neighbors.batch_query(queries.cpu(), supports.cpu(), q_batches.cpu(), s_batches.cpu(), radius=radius)
    if max_neighbors > 0:
        return torch.from_numpy(neighbors[:, :max_neighbors]).to(device)
    else:
        return torch.from_numpy(neighbors).to(device)

def unpack_overlappredator_data(data):
    """Unpack data to get points and features."""
    src_pcd = data['inputs']['src_pc']['pos']
    tgt_pcd = data['inputs']['tgt_pc']['pos']
    src_feats = data['inputs']['src_pc']['feat']
    tgt_feats = data['inputs']['tgt_pc']['feat']
    rot = data['labels']['transform'][:3, :3]
    trans = data['labels']['transform'][:3, 3]
    matching_inds = data['inputs']['correspondences']
    src_pcd_raw = data['inputs']['src_pc']['pos']
    tgt_pcd_raw = data['inputs']['tgt_pc']['pos']
    sample = None

    # Prepare batched data
    batched_points = torch.cat([src_pcd, tgt_pcd], dim=0)
    batched_features = torch.cat([src_feats, tgt_feats], dim=0)
    batched_lengths = torch.tensor([len(src_pcd), len(tgt_pcd)], dtype=torch.int64, device=batched_points.device)

    return {
        'points': batched_points,
        'features': batched_features,
        'lengths': batched_lengths,
        'rot': rot,
        'trans': trans,
        'correspondences': matching_inds,
        'src_pcd_raw': src_pcd_raw,
        'tgt_pcd_raw': tgt_pcd_raw,
        'sample': sample,
        'src_pcd': src_pcd,
        'tgt_pcd': tgt_pcd,
    }


def create_overlappredator_architecture(config, neighborhood_limits):
    """Create architecture for pcr_collator."""
    architecture = []
    r_normal = config.first_subsampling_dl * config.conv_radius
    layer = 0

    for block_i, block in enumerate(config.architecture):
        # Stop when meeting a global pooling or upsampling
        if 'global' in block or 'upsample' in block:
            break

        # Handle deformable blocks
        if 'deformable' in block:
            radius = r_normal * config.deform_radius / config.conv_radius
        else:
            radius = r_normal

        # Add block to architecture
        architecture.append({
            'type': block,
            'radius': radius,
            'sample_dl': 2 * r_normal / config.conv_radius if 'pool' in block or 'strided' in block else r_normal,
            'neighborhood_limit': neighborhood_limits[layer]
        })

        # Update radius for next layer
        if 'pool' in block or 'strided' in block:
            r_normal *= 2
            layer += 1

    return architecture


def pack_overlappredator_results(collated_data, unpacked_data):
    """Pack pcr_collator results into overlappredator format."""
    return {
        'points': collated_data['points'],
        'neighbors': collated_data['neighbors'],
        'pools': collated_data['downsamples'],  # Map downsamples to pools
        'upsamples': collated_data['upsamples'],
        'features': unpacked_data['features'].float(),
        'stack_lengths': collated_data['lengths'],  # Map lengths to stack_lengths
        'rot': unpacked_data['rot'],
        'trans': unpacked_data['trans'],
        'correspondences': unpacked_data['correspondences'],
        'src_pcd_raw': unpacked_data['src_pcd_raw'],
        'tgt_pcd_raw': unpacked_data['tgt_pcd_raw'],
        'sample': unpacked_data['sample'],
    }


def overlappredator_collate_fn(list_data, config, neighborhood_limits):
    assert len(list_data) == 1
    data = list_data[0]  # Get the single item directly

    # Unpack data
    unpacked_data = unpack_overlappredator_data(data)

    # Create architecture
    architecture = create_overlappredator_architecture(config, neighborhood_limits)

    # Call pcr_collator
    collated_data = pcr_collate_fn(
        unpacked_data['points'], unpacked_data['points'],  # Use same points for src and tgt
        architecture,
        downsample_fn=batch_grid_subsampling_kpconv,
        neighbor_fn=batch_neighbors_kpconv,
    )

    # Pack results
    inputs = pack_overlappredator_results(collated_data, unpacked_data)

    # Prepare labels
    labels = {
        'src_pc': unpacked_data['src_pcd'],
        'tgt_pc': unpacked_data['tgt_pcd'],
        'correspondences': unpacked_data['correspondences'],
        'rot': unpacked_data['rot'],
        'trans': unpacked_data['trans'],
    }

    return {'inputs': inputs, 'labels': labels, 'meta_info': {}}
