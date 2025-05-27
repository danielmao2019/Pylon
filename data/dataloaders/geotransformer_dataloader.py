from typing import List
from functools import partial
import math
import torch
from data.dataloaders.base_dataloader import BaseDataLoader
from data.collators.geotransformer.geotransformer_collate_fn import geotransformer_collate_fn


def calibrate_neighbors_stack_mode(
    dataset, collate_fn, num_stages, voxel_size, search_radius, keep_ratio, sample_threshold,
) -> List[int]:
    # Compute higher bound of neighbors number in a neighborhood
    hist_n = math.ceil(4 / 3 * math.pi * (search_radius / voxel_size + 1) ** 3)
    neighbor_hists = torch.zeros((num_stages, hist_n), dtype=torch.int64, device=dataset.device)
    max_neighbor_limits = [hist_n] * num_stages

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):
        data_dict = collate_fn(
            [dataset[i]], num_stages, voxel_size, search_radius, max_neighbor_limits, precompute_data=True
        )

        # update histogram
        counts = [
            torch.sum(neighbors < neighbors.shape[0], dim=1)
            for neighbors in data_dict['inputs']['neighbors']
        ]
        hists = [
            torch.bincount(c, minlength=hist_n)[:hist_n]
            for c in counts
        ]
        neighbor_hists += torch.stack(hists)

        if torch.min(torch.sum(neighbor_hists, dim=1)) > sample_threshold:
            break

    # Calculate neighbor limits using PyTorch operations
    cum_sum = torch.cumsum(neighbor_hists.t(), dim=0)  # (hist_n, num_stages)
    neighbor_limits = torch.sum((cum_sum < (keep_ratio * cum_sum[hist_n - 1, :])).to(torch.int64), dim=0)  # (num_stages,)
    assert torch.all(neighbor_limits > 0), f"{neighbor_limits=}, {cum_sum=}"
    neighbor_limits = neighbor_limits.tolist()
    return neighbor_limits


class GeoTransformerDataloader(BaseDataLoader):

    def __init__(
        self,
        dataset,
        num_stages,
        voxel_size,
        search_radius,
        keep_ratio=0.8,
        sample_threshold=2000,
        **kwargs,
    ) -> None:
        assert 'collate_fn' not in kwargs, 'collate_fn is not allowed to be set'
        self.neighbor_limits = calibrate_neighbors_stack_mode(
            dataset,
            geotransformer_collate_fn,
            num_stages,
            voxel_size,
            search_radius,
            keep_ratio=keep_ratio,
            sample_threshold=sample_threshold,
        )
        super(GeoTransformerDataloader, self).__init__(
            dataset=dataset,
            collate_fn=partial(
                geotransformer_collate_fn,
                num_stages=num_stages,
                voxel_size=voxel_size,
                search_radius=search_radius,
                neighbor_limits=self.neighbor_limits,
            ),
            **kwargs,
        )
