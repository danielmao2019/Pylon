from functools import partial
import numpy as np
import torch
from models.point_cloud_registration.buffer.point_learner import architecture
from data.collators.buffer.buffer_collate_fn import buffer_collate_fn
from data.dataloaders.base_dataloader import BaseDataLoader


num_layer = 1
for block_i, block in enumerate(architecture):
    if ('pool' in block or 'strided' in block):
        num_layer = num_layer + 1


def calibrate_neighbors(dataset, config, collate_fn, keep_ratio=0.8, samples_threshold=2000):
    # From config parameter, compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (config.point.conv_radius) ** 3))
    neighb_hists = np.zeros((num_layer, hist_n), dtype=np.int32)

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):
        batched_input = collate_fn([dataset[i]], config, neighborhood_limits=[hist_n] * 5)

        # update histogram
        counts = [torch.sum(neighb_mat < neighb_mat.shape[0], dim=1).numpy() for neighb_mat in
                  batched_input['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighb_hists += np.vstack(hists)

        if np.min(np.sum(neighb_hists, axis=1)) > samples_threshold:
            break

    cumsum = np.cumsum(neighb_hists.T, axis=0)
    percentiles = np.sum(cumsum < (keep_ratio * cumsum[hist_n - 1, :]), axis=0)

    neighborhood_limits = percentiles
    print(f'neighborhood_limits: {neighborhood_limits}\n')

    return neighborhood_limits


class BufferDataloader(BaseDataLoader):

    def __init__(self, dataset, config, **kwargs) -> None:
        assert 'collate_fn' not in kwargs, 'collate_fn is not allowed to be set'
        self.neighbor_limits = calibrate_neighbors(
            dataset,
            config,
            buffer_collate_fn,
        )
        super(BufferDataloader, self).__init__(
            dataset=dataset,
            collate_fn=partial(
                buffer_collate_fn,
                config=config,
                neighborhood_limits=self.neighbor_limits,
            ),
            **kwargs,
        )
