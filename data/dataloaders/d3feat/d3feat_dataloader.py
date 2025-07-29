from typing import List, Any, Callable
import numpy as np
from functools import partial
import torch
from data.collators.d3feat.d3feat_collate_fn import d3feat_collate_fn
from data.dataloaders.base_dataloader import BaseDataLoader


def calibrate_neighbors(dataset: Any, config: Any, collate_fn: Callable, keep_ratio: float = 0.8, samples_threshold: int = 2000) -> List[int]:

    # From config parameter, compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (config.deform_radius + 1) ** 3))
    neighb_hists = np.zeros((config.num_layers, hist_n), dtype=np.int32)

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):
        batched_dp = collate_fn([dataset[i]], config, neighborhood_limits=[hist_n] * config.num_layers)

        # update histogram
        counts = [torch.sum(neighb_mat < neighb_mat.shape[0], dim=1).cpu().numpy() for neighb_mat in batched_dp['inputs']['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighb_hists += np.vstack(hists)

        if np.min(np.sum(neighb_hists, axis=1)) > samples_threshold:
            break

    cumsum = np.cumsum(neighb_hists.T, axis=0)
    percentiles = np.sum(cumsum < (keep_ratio * cumsum[hist_n - 1, :]), axis=0)

    neighborhood_limits = percentiles.tolist()
    print('\n')

    return neighborhood_limits


class D3FeatDataLoader(BaseDataLoader):
    def __init__(self, dataset: Any, config: Any, **kwargs: Any) -> None:
        assert isinstance(config, dict), 'config must be a dict'
        from easydict import EasyDict
        config = EasyDict(config)
        assert 'collate_fn' not in kwargs, 'collate_fn is not allowed to be set'
        
        # Add architecture configuration like in original D3Feat
        config.architecture = ['simple', 'resnetb']
        for i in range(config.num_layers-1):
            config.architecture.append('resnetb_strided')
            config.architecture.append('resnetb')
            config.architecture.append('resnetb')
        for i in range(config.num_layers-2):
            config.architecture.append('nearest_upsample')
            config.architecture.append('unary')
        config.architecture.append('nearest_upsample')
        config.architecture.append('last_unary')
        
        neighborhood_limits = calibrate_neighbors(
            dataset,
            config,
            collate_fn=d3feat_collate_fn,
        )
        print("neighborhood:", neighborhood_limits)
        super(D3FeatDataLoader, self).__init__(
            dataset=dataset,
            collate_fn=partial(
                d3feat_collate_fn,
                config=config,
                neighborhood_limits=neighborhood_limits,
            ),
            **kwargs,
        )