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
    batched_points_list = []
    batched_features_list = []
    batched_lengths_list = []
    assert len(list_data) == 1
    
    # Initialize correspondences and distance matrices
    sel_corr = np.empty((0, 2), dtype=np.int64)
    dist_keypts = np.empty((0, 0), dtype=np.float32)
    
    # Handle Pylon format data
    datapoint = list_data[0]
    inputs = datapoint['inputs']
    src_pc = inputs['src_pc']
    tgt_pc = inputs['tgt_pc']
    
    # Convert to numpy and extract original format data
    pts0 = src_pc['pos'].detach().cpu().numpy().astype(np.float32)
    pts1 = tgt_pc['pos'].detach().cpu().numpy().astype(np.float32)
    feat0 = src_pc['feat'].detach().cpu().numpy().astype(np.float32)
    feat1 = tgt_pc['feat'].detach().cpu().numpy().astype(np.float32)
    
    # Get correspondences if available
    if 'correspondences' in inputs:
        sel_corr = inputs['correspondences'].detach().cpu().numpy().astype(np.int64)
        
        # Filter out invalid correspondences (indices out of bounds)
        src_size = pts0.shape[0]
        tgt_size = pts1.shape[0]
        valid_mask = (sel_corr[:, 0] < src_size) & (sel_corr[:, 1] < tgt_size)
        sel_corr = sel_corr[valid_mask]
        
        # Limit correspondences for memory efficiency during calibration
        max_corr_for_calibration = 1000
        if sel_corr.shape[0] > max_corr_for_calibration:
            # Randomly sample correspondences to avoid memory issues
            import numpy.random as np_random
            indices = np_random.choice(sel_corr.shape[0], max_corr_for_calibration, replace=False)
            sel_corr = sel_corr[indices]
        
        # Compute distance matrix from correspondences
        if sel_corr.shape[0] > 0:
            corr_pts_src = pts0[sel_corr[:, 0]]
            from scipy.spatial.distance import cdist
            dist_keypts = cdist(corr_pts_src, corr_pts_src).astype(np.float32)
        else:
            dist_keypts = np.empty((0, 0), dtype=np.float32)
    
    batched_points_list.append(pts0)
    batched_points_list.append(pts1)
    batched_features_list.append(feat0)
    batched_features_list.append(feat1)
    batched_lengths_list.append(len(pts0))
    batched_lengths_list.append(len(pts1))
    
    # Get device from original tensors
    device = src_pc['pos'].device
    
    batched_features = torch.from_numpy(np.concatenate(batched_features_list, axis=0)).to(device)
    batched_points = torch.from_numpy(np.concatenate(batched_points_list, axis=0)).to(device)
    batched_lengths = torch.from_numpy(np.array(batched_lengths_list)).int().to(device)

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
        'corr': torch.from_numpy(sel_corr).to(device),
        'dist_keypts': torch.from_numpy(dist_keypts).to(device),
    }

    # Return in Pylon format
    labels = dict(datapoint['labels'])
    labels['correspondences'] = torch.from_numpy(sel_corr).to(device)
    labels['dist_keypts'] = torch.from_numpy(dist_keypts).to(device)
    
    result = {
        'inputs': dict_inputs,
        'labels': labels,
        'meta_info': datapoint['meta_info']
    }
    
    return result