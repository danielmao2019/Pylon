import torch
import numpy as np
import data.collators.d3feat.cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import data.collators.d3feat.cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors


def batch_grid_subsampling_kpconv(points, batches_len, features=None, labels=None, sampleDl=0.1, max_p=0, verbose=0, random_grid_orient=True):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    """
    device = points.device
    
    if (features is None) and (labels is None):
        s_points, s_len = cpp_subsampling.subsample_batch(points.detach().cpu().numpy().astype(np.float32),
                                                          batches_len.detach().cpu().numpy().astype(np.int32),
                                                          sampleDl=sampleDl,
                                                          max_p=max_p,
                                                          verbose=verbose)
        return torch.from_numpy(s_points).to(device), torch.from_numpy(s_len).to(device)

    elif (labels is None):
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(points.detach().cpu().numpy().astype(np.float32),
                                                                      batches_len.detach().cpu().numpy().astype(np.int32),
                                                                      features=features.detach().cpu().numpy().astype(np.float32),
                                                                      sampleDl=sampleDl,
                                                                      max_p=max_p,
                                                                      verbose=verbose)
        return torch.from_numpy(s_points).to(device), torch.from_numpy(s_len).to(device), torch.from_numpy(s_features).to(device)

    elif (features is None):
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(points.detach().cpu().numpy().astype(np.float32),
                                                                    batches_len.detach().cpu().numpy().astype(np.int32),
                                                                    classes=labels.detach().cpu().numpy().astype(np.int32),
                                                                    sampleDl=sampleDl,
                                                                    max_p=max_p,
                                                                    verbose=verbose)
        return torch.from_numpy(s_points).to(device), torch.from_numpy(s_len).to(device), torch.from_numpy(s_labels).to(device)

    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(points.detach().cpu().numpy().astype(np.float32),
                                                                              batches_len.detach().cpu().numpy().astype(np.int32),
                                                                              features=features.detach().cpu().numpy().astype(np.float32),
                                                                              classes=labels.detach().cpu().numpy().astype(np.int32),
                                                                              sampleDl=sampleDl,
                                                                              max_p=max_p,
                                                                              verbose=verbose)
        return torch.from_numpy(s_points).to(device), torch.from_numpy(s_len).to(device), torch.from_numpy(s_features).to(device), torch.from_numpy(s_labels).to(device)


def batch_neighbors_kpconv(queries, supports, q_batches, s_batches, radius, max_neighbors):
    """
    Computes neighbors for a batch of queries and supports
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """
    device = queries.device
    
    neighbors = cpp_neighbors.batch_query(queries.detach().cpu().numpy().astype(np.float32), 
                                         supports.detach().cpu().numpy().astype(np.float32), 
                                         q_batches.detach().cpu().numpy().astype(np.int32), 
                                         s_batches.detach().cpu().numpy().astype(np.int32), 
                                         radius=radius)
    if max_neighbors > 0:
        return torch.from_numpy(neighbors[:, :max_neighbors]).to(device)
    else:
        return torch.from_numpy(neighbors).to(device)


def d3feat_collate_fn(list_data, config, neighborhood_limits):
    """D3Feat collate function for descriptor training."""
    assert len(list_data) == 1
    
    # Handle Pylon format data
    datapoint = list_data[0]
    inputs = datapoint['inputs']
    src_pc = inputs['src_pc']
    tgt_pc = inputs['tgt_pc']
    
    # Keep tensors on device throughout processing
    pts0 = src_pc['pos']  # [N_src, 3] on device
    pts1 = tgt_pc['pos']  # [N_tgt, 3] on device
    feat0 = src_pc['feat']  # [N_src, feat_dim] on device
    feat1 = tgt_pc['feat']  # [N_tgt, feat_dim] on device
    
    device = pts0.device
    
    # Get correspondences and compute distance matrix (on device)
    if 'correspondences' in inputs:
        correspondences = inputs['correspondences']
        
        # Filter out invalid correspondences (indices out of bounds)
        src_size = pts0.shape[0]
        tgt_size = pts1.shape[0]
        valid_mask = (correspondences[:, 0] < src_size) & (correspondences[:, 1] < tgt_size)
        correspondences = correspondences[valid_mask]
        
        # Limit correspondences for memory efficiency during calibration
        max_corr_for_calibration = 1000
        if correspondences.shape[0] > max_corr_for_calibration:
            # Randomly sample correspondences to avoid memory issues
            indices = torch.randperm(correspondences.shape[0], device=device)[:max_corr_for_calibration]
            correspondences = correspondences[indices]
        
        # Compute distance matrix from correspondences (keep on device)
        if correspondences.shape[0] > 0:
            corr_pts_src = pts0[correspondences[:, 0]]
            dist_keypts = torch.cdist(corr_pts_src, corr_pts_src)  # [K, K] on device
        else:
            dist_keypts = torch.empty((0, 0), dtype=torch.float32, device=device)
            
        sel_corr = correspondences
    else:
        sel_corr = torch.empty((0, 2), dtype=torch.long, device=device)
        dist_keypts = torch.empty((0, 0), dtype=torch.float32, device=device)
    
    # Batch points and features (keep on device)
    batched_points = torch.cat([pts0, pts1], dim=0).float()  # [N_total, 3]
    batched_features = torch.cat([feat0, feat1], dim=0).float()  # [N_total, feat_dim]
    batched_lengths = torch.tensor([pts0.shape[0], pts1.shape[0]], dtype=torch.int32, device=device)

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
            conv_i = torch.zeros((0, 1), dtype=torch.int64, device=device)

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
            pool_i = torch.zeros((0, 1), dtype=torch.int64, device=device)
            pool_p = torch.zeros((0, 3), dtype=torch.float32, device=device)
            pool_b = torch.zeros((0,), dtype=torch.int64, device=device)
            up_i = torch.zeros((0, 1), dtype=torch.int64, device=device)

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
    dict_inputs = {
        'points': input_points,
        'neighbors': input_neighbors,
        'pools': input_pools,
        'upsamples': input_upsamples,
        'features': batched_features.float(),
        'stack_lengths': input_batches_len,
        'corr': sel_corr,
        'dist_keypts': dist_keypts,
    }

    # Return in Pylon format
    labels = dict(datapoint['labels'])
    labels['correspondences'] = sel_corr
    labels['dist_keypts'] = dist_keypts
    
    result = {
        'inputs': dict_inputs,
        'labels': labels,
        'meta_info': datapoint['meta_info']
    }
    
    return result