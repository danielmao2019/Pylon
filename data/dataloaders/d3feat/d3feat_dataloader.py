"""
D3Feat DataLoader for Pylon Integration.

This module provides dataloader functionality for D3Feat including neighborhood calibration.
"""

from typing import List, Any, Callable
from functools import partial
import numpy as np
import torch
from data.collators.d3feat.d3feat_collator import collate_fn_descriptor
from data.dataloaders.base_dataloader import BaseDataLoader
from models.point_cloud_registration.d3feat.utils.timer import Timer


def calibrate_neighbors(dataset: Any, config: Any, collate_fn: Callable, keep_ratio: float = 0.8, samples_threshold: int = 2000) -> List[int]:
    """Calibrate neighborhood limits for D3Feat KPConv operations.
    
    Args:
        dataset: The dataset to calibrate on
        config: D3Feat configuration object
        collate_fn: Collate function to use
        keep_ratio: Ratio of neighbors to keep (default: 0.8)
        samples_threshold: Minimum samples per layer (default: 2000)
        
    Returns:
        List of neighborhood limits per layer
    """
    timer = Timer()
    last_display = timer.total_time

    # From config parameter, compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (config.deform_radius + 1) ** 3))
    neighb_hists = np.zeros((config.num_layers, hist_n), dtype=np.int32)

    # Get histogram of neighborhood sizes in 1 epoch max
    for i in range(len(dataset)):
        timer.tic()
        batched_input = collate_fn([dataset[i]], config, neighborhood_limits=[hist_n] * config.num_layers)

        # Update histogram
        counts = [torch.sum(neighb_mat < neighb_mat.shape[0], dim=1).numpy() for neighb_mat in batched_input['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighb_hists += np.vstack(hists)
        timer.toc()

        if timer.total_time - last_display > 0.1:
            last_display = timer.total_time
            print(f"Calib Neighbors {i:08d}: timings {timer.total_time:4.2f}s")

        if np.min(np.sum(neighb_hists, axis=1)) > samples_threshold:
            break

    cumsum = np.cumsum(neighb_hists.T, axis=0)
    percentiles = np.sum(cumsum < (keep_ratio * cumsum[hist_n - 1, :]), axis=0)

    neighborhood_limits = percentiles
    print('\n')

    return neighborhood_limits


class D3FeatDataLoader(BaseDataLoader):
    """D3Feat DataLoader with neighborhood calibration."""
    
    def __init__(
        self,
        dataset: Any,
        batch_size: int = 1,
        num_workers: int = 4,
        shuffle: bool = True,
        neighborhood_limits: List[int] = None,
        keep_ratio: float = 0.8,
        samples_threshold: int = 2000,
        **kwargs
    ):
        """Initialize D3Feat DataLoader.
        
        Args:
            dataset: The dataset to load from
            batch_size: Batch size (default: 1 for D3Feat)
            num_workers: Number of worker processes
            shuffle: Whether to shuffle the data
            neighborhood_limits: Pre-computed neighborhood limits
            keep_ratio: Ratio for neighborhood calibration
            samples_threshold: Threshold for neighborhood calibration
        """
        # Calibrate neighborhood limits if not provided
        if neighborhood_limits is None:
            neighborhood_limits = calibrate_neighbors(
                dataset=dataset, 
                config=dataset.config, 
                collate_fn=collate_fn_descriptor,
                keep_ratio=keep_ratio,
                samples_threshold=samples_threshold
            )
        
        print("neighborhood:", neighborhood_limits)
        
        # Create collate function with calibrated limits
        collate_fn = partial(
            collate_fn_descriptor, 
            config=dataset.config, 
            neighborhood_limits=neighborhood_limits
        )
        
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=False,
            **kwargs
        )
        
        self.neighborhood_limits = neighborhood_limits


def get_d3feat_dataloader(
    dataset: Any, 
    batch_size: int = 1, 
    num_workers: int = 4, 
    shuffle: bool = True, 
    neighborhood_limits: List[int] = None,
    **kwargs
) -> tuple[D3FeatDataLoader, List[int]]:
    """Get D3Feat dataloader with calibrated neighborhood limits.
    
    Args:
        dataset: The dataset to load from
        batch_size: Batch size
        num_workers: Number of worker processes  
        shuffle: Whether to shuffle data
        neighborhood_limits: Pre-computed limits
        
    Returns:
        Tuple of (dataloader, neighborhood_limits)
    """
    dataloader = D3FeatDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        neighborhood_limits=neighborhood_limits,
        **kwargs
    )
    
    return dataloader, dataloader.neighborhood_limits