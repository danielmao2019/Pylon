import numpy as np
import torch
import data.collators.overlappredator.cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import data.collators.overlappredator.cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors


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

def overlappredator_collate_fn(list_data, config, neighborhood_limits):
    batched_points_list = []
    batched_features_list = []
    batched_lengths_list = []
    assert len(list_data) == 1

    for ind, dp in enumerate(list_data):
        # unpack
        src_pcd = dp['inputs']['src_pc']['pos']
        tgt_pcd = dp['inputs']['tgt_pc']['pos']
        src_feats = dp['inputs']['src_pc']['feat']
        tgt_feats = dp['inputs']['tgt_pc']['feat']
        rot = dp['labels']['transform'][:3, :3]
        trans = dp['labels']['transform'][:3, 3]
        matching_inds = dp['inputs']['correspondences']
        src_pcd_raw = dp['inputs']['src_pc']['pos']
        tgt_pcd_raw = dp['inputs']['tgt_pc']['pos']
        sample = None

        batched_points_list.append(src_pcd)
        batched_points_list.append(tgt_pcd)
        batched_features_list.append(src_feats)
        batched_features_list.append(tgt_feats)
        batched_lengths_list.append(len(src_pcd))
        batched_lengths_list.append(len(tgt_pcd))

    batched_features = torch.cat(batched_features_list, dim=0)
    batched_points = torch.cat(batched_points_list, dim=0)
    batched_lengths = torch.tensor(batched_lengths_list, dtype=torch.int64, device=batched_points.device)

    # Starting radius of convolutions
    r_normal = config.first_subsampling_dl * config.conv_radius

    # Starting layer
    layer_blocks = []
    layer = 0

    # Lists of inputs
    input_points = []
    input_neighbors = []
    input_pools = []
    input_upsamples = []
    input_batches_len = []

    for block_i, block in enumerate(config.architecture):

        # Stop when meeting a global pooling or upsampling
        if 'global' in block or 'upsample' in block:
            break

        # Get all blocks of the layer
        if not ('pool' in block or 'strided' in block):
            layer_blocks += [block]
            if block_i < len(config.architecture) - 1 and not ('upsample' in config.architecture[block_i + 1]):
                continue

        # Convolution neighbors indices
        # *****************************

        if layer_blocks:
            # Convolutions are done in this layer, compute the neighbors with the good radius
            if np.any(['deformable' in blck for blck in layer_blocks[:-1]]):
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal
            conv_i = batch_neighbors_kpconv(batched_points, batched_points, batched_lengths, batched_lengths, r, neighborhood_limits[layer])

        else:
            # This layer only perform pooling, no neighbors required
            conv_i = torch.zeros((0, 1), dtype=torch.int64)

        # Pooling neighbors indices
        # *************************

        # If end of layer is a pooling operation
        if 'pool' in block or 'strided' in block:

            # New subsampling length
            dl = 2 * r_normal / config.conv_radius

            # Subsampled points
            pool_p, pool_b = batch_grid_subsampling_kpconv(batched_points, batched_lengths, sampleDl=dl)

            # Radius of pooled neighbors
            if 'deformable' in block:
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal

            # Subsample indices
            pool_i = batch_neighbors_kpconv(pool_p, batched_points, pool_b, batched_lengths, r, neighborhood_limits[layer])

            # Upsample indices (with the radius of the next layer to keep wanted density)
            up_i = batch_neighbors_kpconv(batched_points, pool_p, batched_lengths, pool_b, 2 * r, neighborhood_limits[layer])

        else:
            # No pooling in the end of this layer, no pooling indices required
            pool_i = torch.zeros((0, 1), dtype=torch.int64)
            pool_p = torch.zeros((0, 3), dtype=torch.float32)
            pool_b = torch.zeros((0,), dtype=torch.int64)
            up_i = torch.zeros((0, 1), dtype=torch.int64)

        # Updating input lists
        input_points += [batched_points.float()]
        input_neighbors += [conv_i.long()]
        input_pools += [pool_i.long()]
        input_upsamples += [up_i.long()]
        input_batches_len += [batched_lengths]

        # New points for next layer
        batched_points = pool_p
        batched_lengths = pool_b

        # Update radius and reset blocks
        r_normal *= 2
        layer += 1
        layer_blocks = []

    ###############
    # Return inputs
    ###############
    inputs = {
        'points': input_points,
        'neighbors': input_neighbors,
        'pools': input_pools,
        'upsamples': input_upsamples,
        'features': batched_features.float(),
        'stack_lengths': input_batches_len,
        'rot': rot,
        'trans': trans,
        'correspondences': matching_inds,
        'src_pcd_raw': src_pcd_raw,
        'tgt_pcd_raw': tgt_pcd_raw,
        'sample': sample,
    }
    labels = {
        'src_pc': src_pcd,
        'tgt_pc': tgt_pcd,
        'correspondences': matching_inds,
        'rot': rot,
        'trans': trans,
    }
    meta_info = {}

    return {'inputs': inputs, 'labels': labels, 'meta_info': meta_info}
