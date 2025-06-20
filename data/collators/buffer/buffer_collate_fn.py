from typing import List, Dict, Any, Optional, Tuple, Union
import torch
import numpy as np
import data.collators.buffer.cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import data.collators.buffer.cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors
from data.collators.pcr_collator import pcr_collate_fn
from models.point_cloud_registration.buffer.point_learner import architecture as _architecture


def batch_grid_subsampling_kpconv(
    points: torch.Tensor, 
    batches_len: torch.Tensor, 
    sampleDl: float, 
    features: Optional[torch.Tensor] = None, 
    labels: Optional[torch.Tensor] = None, 
    max_p: int = 0, 
    verbose: int = 0
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
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


def batch_neighbors_kpconv(
    queries: torch.Tensor, 
    supports: torch.Tensor, 
    q_batches: torch.Tensor, 
    s_batches: torch.Tensor, 
    radius: float, 
    max_neighbors: int
) -> torch.Tensor:
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


def unpack_buffer_data(data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Unpack data to get points and features."""
    src_fds_points = data['inputs']['src_pc_fds']['pos']
    tgt_fds_points = data['inputs']['tgt_pc_fds']['pos']
    src_sds_points = data['inputs']['src_pc_sds']['pos']
    tgt_sds_points = data['inputs']['tgt_pc_sds']['pos']
    src_features = data['inputs']['src_pc_sds']['normals']
    tgt_features = data['inputs']['tgt_pc_sds']['normals']
    relt_pose = data['labels']['transform']

    # Prepare batched data
    batched_points = torch.cat([src_sds_points, tgt_sds_points], dim=0)
    batched_features = torch.cat([src_features, tgt_features], dim=0)
    batched_lengths = torch.tensor([len(src_sds_points), len(tgt_sds_points)], dtype=torch.int64, device=batched_points.device)

    return {
        'src_points': src_sds_points,
        'tgt_points': tgt_sds_points,
        'lengths': batched_lengths,
        'features': batched_features,
        'src_pcd_raw': src_fds_points,
        'tgt_pcd_raw': tgt_fds_points,
        'src_pcd': src_sds_points,
        'tgt_pcd': tgt_sds_points,
        'relt_pose': relt_pose,
    }


def create_buffer_architecture(config: Any, neighborhood_limits: List[int]) -> List[Dict[str, Any]]:
    """Create architecture for pcr_collator."""
    architecture = []
    r_normal = config.data.voxel_size_0 * config.point.conv_radius
    layer_blocks = []
    layer = 0

    for block_i, block in enumerate(_architecture):
        # Stop when meeting a global pooling or upsampling
        if 'global' in block or 'upsample' in block:
            break

        # Get all blocks of the layer
        if not ('pool' in block or 'strided' in block):
            layer_blocks += [block]
            if block_i < len(_architecture) - 1 and not ('upsample' in _architecture[block_i + 1]):
                continue

        # Add block to architecture
        architecture.append({
            'neighbor': layer_blocks,
            'neighbor_radius': r_normal,
            'neighbor_neighborhood_limit': neighborhood_limits[layer],
            'downsample': 'pool' in block or 'strided' in block,
            'sample_dl': 2 * r_normal / config.point.conv_radius,
            'downsample_radius': r_normal,
            'downsample_neighborhood_limit': neighborhood_limits[layer],
            'upsample_radius': 2 * r_normal,
            'upsample_neighborhood_limit': neighborhood_limits[layer],
        })

        r_normal *= 2
        layer += 1
        layer_blocks = []

    # This isn't true because we first use uncalibrated neighborhood limits to 
    # compute calibrated neighborhood limits, and then use the calibrated neighborhood
    # limits to do collation for the dataloader.
    # assert len(architecture) == len(neighborhood_limits)
    return architecture


def pack_buffer_results(collated_data: Dict[str, List[torch.Tensor]], unpacked_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Pack pcr_collator results into buffer format."""
    return {
        'points': collated_data['points'],
        'stack_lengths': collated_data['lengths'],  # Map lengths to stack_lengths
        'neighbors': collated_data['neighbors'],
        'pools': collated_data['downsamples'],  # Map downsamples to pools
        'upsamples': collated_data['upsamples'],
        'features': unpacked_data['features'].float(),
        'src_pcd_raw': unpacked_data['src_pcd_raw'],
        'tgt_pcd_raw': unpacked_data['tgt_pcd_raw'],
        'src_pcd': unpacked_data['src_pcd'],
        'tgt_pcd': unpacked_data['tgt_pcd'],
        'relt_pose': unpacked_data['relt_pose'],
    }


def buffer_collate_fn(list_data: List[Dict[str, Any]], config: Any, neighborhood_limits: List[int]) -> Dict[str, Any]:
    assert len(list_data) == 1
    data = list_data[0]  # Get the single item directly

    # Unpack data
    unpacked_data = unpack_buffer_data(data)

    # Create architecture
    architecture = create_buffer_architecture(config, neighborhood_limits)

    # Call pcr_collator
    collated_data = pcr_collate_fn(
        src_points=unpacked_data['src_points'],
        tgt_points=unpacked_data['tgt_points'],
        architecture=architecture,
        downsample_fn=batch_grid_subsampling_kpconv,
        neighbor_fn=batch_neighbors_kpconv,
    )

    # Pack results
    dict_inputs = pack_buffer_results(collated_data, unpacked_data)

    return {
        'inputs': dict_inputs,
        'labels': data['labels'],
        'meta_info': data['meta_info'],
    }
