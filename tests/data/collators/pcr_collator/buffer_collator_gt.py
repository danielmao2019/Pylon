import torch
from data.collators.buffer.buffer_collate_fn import batch_grid_subsampling_kpconv, batch_neighbors_kpconv
from models.point_cloud_registration.buffer.point_learner import architecture


def buffer_collate_fn_gt(list_data, config, neighborhood_limits):
    batched_points_list = []
    batched_lengths_list = []
    batched_features_list = []# = np.ones_like(input_points[0][:, :0]).astype(np.float32)
    assert len(list_data) == 1
    list_data = list_data[0]

    s_pts, t_pts = list_data['inputs']['src_pc_fds']['pos'], list_data['inputs']['tgt_pc_fds']['pos']
    relt_pose = list_data['labels']['transform']
    s_kpt, t_kpt = list_data['inputs']['src_pc_sds'], list_data['inputs']['tgt_pc_sds']
    src_kpt = s_kpt['pos']
    tgt_kpt = t_kpt['pos']
    src_f = s_kpt['normals']
    tgt_f = t_kpt['normals']
    batched_points_list.append(src_kpt)
    batched_points_list.append(tgt_kpt)
    batched_features_list.append(src_f)
    batched_features_list.append(tgt_f)
    batched_lengths_list.append(len(src_kpt))
    batched_lengths_list.append(len(tgt_kpt))

    batched_points = torch.cat(batched_points_list, dim=0)
    batched_features = torch.cat(batched_features_list, dim=0)
    batched_lengths = torch.tensor(batched_lengths_list, dtype=torch.int64, device=batched_points.device)

    # Starting radius of convolutions
    r_normal = config.data.voxel_size_0 * config.point.conv_radius

    # Starting layer
    layer_blocks = []
    layer = 0

    # Lists of inputs
    input_points = []
    input_neighbors = []
    input_pools = []
    input_upsamples = []
    input_batches_len = []

    for block_i, block in enumerate(architecture):

        # Stop when meeting a global pooling or upsampling
        if 'global' in block or 'upsample' in block:
            break

        # Get all blocks of the layer
        if not ('pool' in block or 'strided' in block):
            layer_blocks += [block]
            if block_i < len(architecture) - 1 and not ('upsample' in architecture[block_i + 1]):
                continue

        # Convolution neighbors indices
        # *****************************

        if layer_blocks:
            # Convolutions are done in this layer, compute the neighbors with the good radius
            r = r_normal
            conv_i = batch_neighbors_kpconv(batched_points, batched_points, batched_lengths, batched_lengths, r,
                                            neighborhood_limits[layer])

        else:
            # This layer only perform pooling, no neighbors required
            conv_i = torch.zeros((0, 1), dtype=torch.int64)

        # Pooling neighbors indices
        # *************************

        # If end of layer is a pooling operation
        if 'pool' in block or 'strided' in block:

            # New subsampling length
            dl = 2 * r_normal / config.point.conv_radius

            # Subsampled points
            pool_p, pool_b = batch_grid_subsampling_kpconv(batched_points, batched_lengths, sampleDl=dl)

            # Radius of pooled neighbors
            r = r_normal

            # Subsample indices
            pool_i = batch_neighbors_kpconv(pool_p, batched_points, pool_b, batched_lengths, r,
                                            neighborhood_limits[layer])

            # Upsample indices (with the radius of the next layer to keep wanted density)
            up_i = batch_neighbors_kpconv(batched_points, pool_p, batched_lengths, pool_b, 2 * r,
                                          neighborhood_limits[layer])

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
        r_normal *= 2.0
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
        'src_pcd_raw': s_pts,
        'tgt_pcd_raw': t_pts,
        'src_pcd': src_kpt,
        'tgt_pcd': tgt_kpt,
        'relt_pose': relt_pose,
    }

    return {
        'inputs': dict_inputs,
        'labels': list_data['labels'],
        'meta_info': list_data['meta_info'],
    }
