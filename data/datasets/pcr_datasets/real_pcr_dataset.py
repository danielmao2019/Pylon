from typing import List, Dict, Any, Tuple
from functools import partial
import os
import multiprocessing
import glob
import json
import numpy as np
import torch
from data.datasets.base_dataset import BaseDataset


class RealPCRDataset(BaseDataset):

    def __init__(self, gt_transforms: str, **kwargs) -> None:
        self.gt_transforms = gt_transforms
        super(RealPCRDataset, self).__init__(**kwargs)

    def _init_annotations(self) -> None:
        self.file_paths = sorted(glob.glob(os.path.join(self.data_root, '*.ply')))
        with open(self.gt_transforms, 'r') as f:
            self.gt_transforms: List[Dict[str, Any]] = json.load(f)
        assert len(self.file_paths) == len(self.gt_transforms)
        assert all(isinstance(transform, dict) for transform in self.gt_transforms)
        assert all(transform.keys() == {'filepath', 'transform'} for transform in self.gt_transforms)
        self.cache_dir = os.path.join(os.path.dirname(self.data_root), 'real_pcr_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_files = sorted(glob.glob(os.path.join(self.cache_dir, 'datapoint_*.pt')))
        if len(cache_files) > 0:
            self.annotations = cache_files
            print(f"Loaded {len(cache_files)} cached datapoints.")
        else:
            print(f"No cached datapoints found in {self.cache_dir}")
            num_workers = max(1, multiprocessing.cpu_count() - 1)
            print(f"Processing point clouds using {num_workers} workers...")

            process_func = partial(
                process_single_point_cloud,
                grid_sampling=self._grid_sampling,
                min_points=self._min_points,
                max_points=self._max_points,
            )

            # Use multiprocessing to process files in parallel with chunksize for better performance
            with multiprocessing.Pool(num_workers) as pool:
                # Use chunksize=1 for better load balancing with varying file sizes
                results = pool.map(process_func, self.file_paths, chunksize=1)

            # Flatten the results list
            self.annotations = [datapoint for sublist in results for datapoint in sublist]

            # Save datapoints to cache in parallel
            print(f"Saving {len(self.annotations)} datapoints to cache...")
            save_args = [(i, datapoint, self.cache_dir) for i, datapoint in enumerate(self.annotations)]
            with multiprocessing.Pool(num_workers) as pool:
                pool.map(save_datapoint, save_args, chunksize=1)
            print(f"Created and cached {len(self.annotations)} datapoints.")

        # Split annotations into train/val/test
        np.random.seed(42)
        indices = np.random.permutation(len(self.annotations))
        train_idx = int(0.7 * len(indices))
        val_idx = int(0.85 * len(indices))  # 70% + 15%

        if self.split == 'train':
            select_indices = indices[:train_idx]
        elif self.split == 'val':
            select_indices = indices[train_idx:val_idx]
        else:  # test
            select_indices = indices[val_idx:]

        # Select annotations for current split
        self.annotations = [self.annotations[i] for i in select_indices]

        # Update dataset size
        self.DATASET_SIZE[self.split] = len(self.annotations)

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        """Load a datapoint using point indices and generate synthetic pair."""
        # Get voxel data
        datapoint_cache = self.annotations[idx]

        # If annotations are filepaths, load the data
        if isinstance(datapoint_cache, str):
            datapoint_cache = torch.load(datapoint_cache)

        src_points = datapoint_cache['src_points']
        tgt_points = datapoint_cache['tgt_points']
        gt_transform = datapoint_cache['transform']

        inputs = {
            'src_pc': {
                'pos': src_points,
                'feat': torch.ones((src_points.shape[0], 1), dtype=torch.float32),
            },
            'tgt_pc': {
                'pos': tgt_points,
                'feat': torch.ones((tgt_points.shape[0], 1), dtype=torch.float32),
            },
        }

        labels = {
            'transform': gt_transform,
        }

        meta_info = {
            'idx': idx,
            'src_path': datapoint_cache['src_path'],
            'tgt_path': datapoint_cache['tgt_path'],
        }

        return inputs, labels, meta_info
