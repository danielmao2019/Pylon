from typing import List
from functools import partial
import numpy as np
import torch
from data.dataloaders.base_dataloader import BaseDataLoader
from data.collators.geotransformer.registration_collate_fn_stack_mode import registration_collate_fn_stack_mode


def calibrate_neighbors_stack_mode(
    dataset, collate_fn, num_stages, voxel_size, search_radius, keep_ratio=0.8, sample_threshold=2000
) -> List[int]:
    # Compute higher bound of neighbors number in a neighborhood
    hist_n = int(torch.ceil(4 / 3 * torch.pi * (search_radius / voxel_size + 1) ** 3))
    neighbor_hists = torch.zeros((num_stages, hist_n), dtype=torch.int64)
    max_neighbor_limits = [hist_n] * num_stages

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):
        data_dict = collate_fn(
            [dataset[i]], num_stages, voxel_size, search_radius, max_neighbor_limits, precompute_data=True
        )

        # update histogram
        # Get neighbors from both source and target point clouds
        src_neighbors = data_dict['inputs']['src_pc']['neighbors']
        tgt_neighbors = data_dict['inputs']['tgt_pc']['neighbors']

        # Count neighbors separately for source and target and add them
        counts = [torch.cat([
            torch.sum(src_neighbor < src_neighbor.shape[0], axis=1),
            torch.sum(tgt_neighbor < tgt_neighbor.shape[0], axis=1),
        ], dim=0) for src_neighbor, tgt_neighbor in zip(src_neighbors, tgt_neighbors)]

        # Create histograms from combined counts
        hists = [torch.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighbor_hists += torch.stack(hists)

        if torch.min(torch.sum(neighbor_hists, dim=1)) > sample_threshold:
            break

    # Calculate neighbor limits using PyTorch operations
    cum_sum = torch.cumsum(neighbor_hists.t(), dim=0)
    neighbor_limits = torch.sum((cum_sum < (keep_ratio * cum_sum[hist_n - 1, :])).to(torch.int64), dim=0)
    neighbor_limits = neighbor_limits.tolist()
    assert all([l > 0 for l in neighbor_limits]), f"{neighbor_limits=}, {cum_sum=}"
    return neighbor_limits


class GeoTransformerDataloader(BaseDataLoader):
    def __init__(self, dataset, num_stages, voxel_size, search_radius, **kwargs):
        assert 'collate_fn' not in kwargs, 'collate_fn is not allowed to be set'
        neighbor_limits = calibrate_neighbors_stack_mode(
            dataset,
            registration_collate_fn_stack_mode,
            num_stages,
            voxel_size,
            search_radius,
        )
        super(GeoTransformerDataloader, self).__init__(
            dataset=dataset,
            collate_fn=partial(
                registration_collate_fn_stack_mode,
                num_stages=num_stages,
                voxel_size=voxel_size,
                search_radius=search_radius,
                neighbor_limits=neighbor_limits,
            ),
            **kwargs,
        )
