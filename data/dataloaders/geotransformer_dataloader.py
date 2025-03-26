from functools import partial

from data.dataloaders.base_dataloader import BaseDataloader
from data.collators.geotransformer.registration_collate_fn_stack_mode import registration_collate_fn_stack_mode


def calibrate_neighbors_stack_mode(
    dataset, collate_fn, num_stages, voxel_size, search_radius, keep_ratio=0.8, sample_threshold=2000
):
    # Compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (search_radius / voxel_size + 1) ** 3))
    neighbor_hists = np.zeros((num_stages, hist_n), dtype=np.int32)
    max_neighbor_limits = [hist_n] * num_stages

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):
        data_dict = collate_fn(
            [dataset[i]], num_stages, voxel_size, search_radius, max_neighbor_limits, precompute_data=True
        )

        # update histogram
        counts = [np.sum(neighbors.numpy() < neighbors.shape[0], axis=1) for neighbors in data_dict['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighbor_hists += np.vstack(hists)

        if np.min(np.sum(neighbor_hists, axis=1)) > sample_threshold:
            break

    cum_sum = np.cumsum(neighbor_hists.T, axis=0)
    neighbor_limits = np.sum(cum_sum < (keep_ratio * cum_sum[hist_n - 1, :]), axis=0)

    return neighbor_limits


class GeoTransformerDataloader(BaseDataloader):
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
