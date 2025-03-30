from typing import List
import numpy as np
from functools import partial
import torch
from data.collators.overlappredator.overlappredator_collate_fn import overlappredator_collate_fn
from data.dataloaders.base_dataloader import BaseDataLoader


def calibrate_neighbors(dataset, config, collate_fn, keep_ratio=0.8, samples_threshold=2000) -> List[int]:

    # From config parameter, compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (config.deform_radius + 1) ** 3))
    neighb_hists = np.zeros((config.num_layers, hist_n), dtype=np.int32)

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):
        batched_input = collate_fn([dataset[i]], config, neighborhood_limits=[hist_n] * 5)

        # update histogram
        counts = [torch.sum(neighb_mat < neighb_mat.shape[0], dim=1).numpy() for neighb_mat in batched_input['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighb_hists += np.vstack(hists)

        if np.min(np.sum(neighb_hists, axis=1)) > samples_threshold:
            break

    cumsum = np.cumsum(neighb_hists.T, axis=0)
    percentiles = np.sum(cumsum < (keep_ratio * cumsum[hist_n - 1, :]), axis=0)

    neighborhood_limits = percentiles.tolist()
    print('\n')

    return neighborhood_limits


class OverlapPredatorDataloader(BaseDataLoader):
    def __init__(self, dataset, config, **kwargs):
        assert isinstance(config, dict), 'config must be a dict'
        from easydict import EasyDict
        config = EasyDict(config)
        assert 'collate_fn' not in kwargs, 'collate_fn is not allowed to be set'
        neighborhood_limits = calibrate_neighbors(
            dataset,
            config,
            collate_fn=overlappredator_collate_fn,
        )
        super(OverlapPredatorDataloader, self).__init__(
            dataset=dataset,
            collate_fn=partial(
                overlappredator_collate_fn,
                config=config,
                neighborhood_limits=neighborhood_limits,
            ),
            **kwargs,
        )
