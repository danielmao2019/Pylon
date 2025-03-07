from typing import Tuple, Dict, Any, Optional
import os
import glob
import random
import numpy as np
import torch
from sklearn.neighbors import KDTree
from utils.torch_points3d import GridSampling3D, CylinderSampling
from data.datasets import BaseDataset
import utils


class Urb3DCDDataset(BaseDataset):
    """Combined class that supports both sphere and cylinder sampling within an area
    during training and validation. Default sampling radius is 2m.
    If sample_per_epoch is not specified, samples are placed on a 2m grid.
    """

    INPUT_NAMES = ['pc_0', 'pc_1', 'kdtree_0', 'kdtree_1']
    LABEL_NAMES = ['change_map']
    NUM_CLASSES = 7
    INV_OBJECT_LABEL = {
        0: "unchanged",
        1: "newlyBuilt",
        2: "deconstructed",
        3: "newVegetation",
        4: "vegetationGrowUp",
        5: "vegetationRemoved",
        6: "mobileObjects"
    }
    CLASS_LABELS = {name: i for i, name in INV_OBJECT_LABEL.items()}
    IGNORE_LABEL = -1
    SPLIT_OPTIONS = {'train', 'val', 'test'}
    SPLIT_MAP = {
        'train': 'TrainLarge-1c',  # Using the largest training set by default
        'val': 'Val',
        'test': 'Test'
    }
    VERSION_MAP = {
        1: {
            'dir': 'IEEE_Dataset_V1',
            'subdir': '1-Lidar05',
            'nameInPly': 'Urb3DSimul'
        },
        2: {
            'dir': 'IEEE_Dataset_V2_Lid05_MS',
            'subdir': 'Lidar05',
            'nameInPly': 'params'
        }
    }

    def __init__(self, sample_per_epoch: int = 100, radius: float = 2, fix_samples: bool = False, version: int = 1, *args, **kwargs) -> None:
        if version not in self.VERSION_MAP:
            raise ValueError(f"Version {version} is not supported. Must be one of {list(self.VERSION_MAP.keys())}")
        
        self._sample_per_epoch = sample_per_epoch
        self._radius = radius
        self.fix_samples = fix_samples
        self.version = version
        self._grid_sampling = GridSampling3D(size=radius / 10.0)  # Renamed to be more generic
        super(Urb3DCDDataset, self).__init__(*args, **kwargs)

    def _init_annotations(self) -> None:
        """Initialize file paths for point clouds."""
        version_info = self.VERSION_MAP[self.version]
        base_dir = os.path.join(self.data_root, version_info['dir'], version_info['subdir'], self.SPLIT_MAP[self.split])

        # Find all point cloud pairs using glob
        pc0_files = sorted(glob.glob(os.path.join(base_dir, "**/pointCloud0.ply"), recursive=True))
        pc1_files = sorted(glob.glob(os.path.join(base_dir, "**/pointCloud1.ply"), recursive=True))

        if len(pc0_files) != len(pc1_files):
            raise ValueError(f"Number of pointCloud0 files ({len(pc0_files)}) does not match pointCloud1 files ({len(pc1_files)})")

        if len(pc0_files) == 0:
            raise ValueError(f"No point cloud pairs found in {base_dir}")

        self.annotations = [
            {'pc_0_filepath': pc0, 'pc_1_filepath': pc1}
            for pc0, pc1 in zip(pc0_files, pc1_files)
        ]
        self._prepare_centers()

    def _load_point_cloud_pair(self, idx: int) -> Dict[str, Any]:
        """Loads a pair of point clouds and builds KDTrees for them."""
        print("Loading " + self.annotations[idx]['pc_1_filepath'])
        nameInPly = self.VERSION_MAP[self.version]['nameInPly']
        
        # Load first point cloud
        pc0 = utils.io.load_point_cloud(self.annotations[idx]['pc_0_filepath'], nameInPly=nameInPly)
        assert pc0.size(1) == 4, f"{pc0.shape=}"
        pc0 = pc0[:, :3]
        
        # Load second point cloud
        pc = utils.io.load_point_cloud(self.annotations[idx]['pc_1_filepath'], nameInPly=nameInPly)
        assert pc.size(1) == 4, f"{pc.shape=}"
        pc1 = pc[:, :3]
        change_map = pc[:, 3]  # Labels should be at the 4th column 0:X 1:Y 2:Z 3:Label
        
        # Convert to correct types
        pc0 = pc0.type(torch.float32)
        pc1 = pc1.type(torch.float32)
        change_map = change_map.type(torch.int64)
        
        # Build KDTrees and return data
        data = {
            'pc_0': pc0,
            'pc_1': pc1,
            'change_map': change_map,
            'kdtree_0': KDTree(np.asarray(pc0), leaf_size=10),
            'kdtree_1': KDTree(np.asarray(pc1), leaf_size=10),
        }
        return data

    def _prepare_centers(self) -> None:
        """Prepares centers based on whether random sampling or grid sampling is used."""
        # Common preparation code
        centers_list = []
        for idx in range(len(self.annotations)):
            data = self._load_point_cloud_pair(idx)
            data_dict = {
                'pos': data['pc_1'],
                'change_map': data['change_map']
            }
            sampled_data = self._grid_sampling(data_dict)
            centers = {
                'pos': sampled_data['pos'],
                'idx': idx * torch.ones(len(sampled_data['pos']), dtype=torch.long),
                'change_map': sampled_data['change_map']
            }
            centers_list.append(centers)

        # Convert to single dictionary with concatenated tensors
        all_centers = {
            'pos': torch.cat([c['pos'] for c in centers_list], dim=0),
            'idx': torch.cat([c['idx'] for c in centers_list], dim=0),
            'change_map': torch.cat([c['change_map'] for c in centers_list], dim=0)
        }

        if self._sample_per_epoch > 0:  # Random/Fixed sampling mode
            self._centers_for_sampling = all_centers
            
            # Calculate label statistics for balanced sampling
            labels, label_counts = np.unique(all_centers['change_map'].numpy(), return_counts=True)
            self._label_counts = np.sqrt(label_counts.mean() / label_counts)
            self._label_counts /= np.sum(self._label_counts)
            self._labels = labels
            self.weight_classes = torch.tensor(self._label_counts, dtype=torch.float32)
            
            if self.fix_samples:
                self._prepare_fixed_sampling()
        else:  # Grid sampling mode
            self.grid_regular_centers = all_centers

    def _prepare_fixed_sampling(self) -> None:
        """Fixes the sampled locations for consistency across epochs."""
        np.random.seed(1)
        chosen_labels = np.random.choice(self._labels, p=self._label_counts, size=(self._sample_per_epoch, 1))
        unique_labels, label_counts = np.unique(chosen_labels, return_counts=True)
        
        self._centers_for_sampling_fixed = []
        for label, count in zip(unique_labels, label_counts):
            valid_centers = self._centers_for_sampling['pos'][self._centers_for_sampling['change_map'] == label]
            selected_idx = np.random.randint(low=0, high=valid_centers.shape[0], size=(count,))
            self._centers_for_sampling_fixed.append(valid_centers[selected_idx])
        
        self._centers_for_sampling_fixed = torch.cat(self._centers_for_sampling_fixed, 0)

    def __len__(self) -> int:
        return self._sample_per_epoch if self._sample_per_epoch > 0 else self.grid_regular_centers['pos'].shape[0]

    def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        """Load a datapoint for the parent class interface."""
        # Get sample based on sampling mode
        if self._sample_per_epoch > 0:
            sample = self._get_fixed_sample(idx) if self.fix_samples else self._get_random()
        else:
            sample = self._get_regular_sample(idx)
            
        if sample is None:
            raise ValueError(f"Failed to load datapoint at index {idx}")
        
        pc0, pc1 = sample['pc_0'], sample['pc_1']
            
        try:
            # Normalize point clouds
            self._normalize(pc0, pc1)
            
            # Prepare return values in format expected by parent class
            inputs = {
                'pc_0': pc0,
                'pc_1': pc1,
                'kdtree_0': self._loaded_data[sample['idx']]['kdtree_0'],
                'kdtree_1': self._loaded_data[sample['idx']]['kdtree_1'],
            }
            labels = {
                'change_map': sample['change_map'],
            }
            meta_info = {
                **self.annotations[sample['idx']],
                'point_idx_pc0': sample['point_idx_pc0'],
                'point_idx_pc1': sample['point_idx_pc1'],
                'idx': sample['idx']
            }
            return inputs, labels, meta_info
        except Exception as e:
            print(f"Normalization failed: {e}")
            print(f"pc_0 shape: {pc0.shape}, pc_1 shape: {pc1.shape}")
            raise

    def _normalize(self, pc0: torch.Tensor, pc1: torch.Tensor) -> None:
        """Normalizes point clouds."""
        min0 = torch.unsqueeze(pc0.min(0)[0], 0)
        min1 = torch.unsqueeze(pc1.min(0)[0], 0)
        minG = torch.cat((min0, min1), axis=0).min(0)[0]
        pc0[:, 0] = (pc0[:, 0] - minG[0])  # x
        pc0[:, 1] = (pc0[:, 1] - minG[1])  # y
        pc0[:, 2] = (pc0[:, 2] - minG[2])  # z
        pc1[:, 0] = (pc1[:, 0] - minG[0])  # x
        pc1[:, 1] = (pc1[:, 1] - minG[1])  # y
        pc1[:, 2] = (pc1[:, 2] - minG[2])  # z

    def _get_fixed_sample(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """Get a sample from fixed sampling locations."""
        if idx >= self._centers_for_sampling_fixed.shape[0]:
            raise ValueError(f"Index {idx} out of range for fixed sampling")
            
        center = self._centers_for_sampling_fixed[idx]
        sample_idx = self._centers_for_sampling['idx'][idx].int()
        
        return self._sample_cylinder(self._loaded_data[sample_idx], center, sample_idx)

    def _get_regular_sample(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """Retrieves a regular sample without normalization."""
        while idx < self.grid_regular_centers['pos'].shape[0]:
            center, sample_idx = self._extract_center_info(self.grid_regular_centers, idx)
            data = self._load_point_cloud_pair(sample_idx)
            
            sample = self._sample_cylinder(data, center, sample_idx, apply_transform=True)

            if sample:
                return sample

            print('pair not correct')
            idx += 1
        return None

    def _get_random(self) -> Optional[Dict[str, torch.Tensor]]:
        """Randomly selects a sample without normalization."""
        chosen_label = np.random.choice(self._labels, p=self._label_counts)
        mask = self._centers_for_sampling['change_map'] == chosen_label
        valid_centers = {
            'pos': self._centers_for_sampling['pos'][mask],
            'idx': self._centers_for_sampling['idx'][mask],
            'change_map': self._centers_for_sampling['change_map'][mask]
        }

        if valid_centers['pos'].shape[0] == 0:
            return None

        center_idx = int(random.random() * (valid_centers['pos'].shape[0] - 1))
        center = valid_centers['pos'][center_idx]
        sample_idx = valid_centers['idx'][center_idx].int()
        data = self._load_point_cloud_pair(sample_idx)
        
        return self._sample_cylinder(data, center, sample_idx, apply_transform=True)

    def _sample_cylinder(self, data: Dict[str, Any], center: torch.Tensor, idx: int, apply_transform: bool = False) -> Dict[str, torch.Tensor]:
        """Applies cylindrical sampling and optional transformations."""
        cylinder_sampler = CylinderSampling(self._radius, center, align_origin=False)
        
        # Create data dictionaries for both point clouds
        data_dict_0 = {'pos': data['pc_0']}
        data_dict_1 = {
            'pos': data['pc_1'],
            'change_map': data['change_map']
        }
        
        # Sample points using the KDTrees
        sampled_data_0 = cylinder_sampler(data['kdtree_0'], data_dict_0)
        sampled_data_1 = cylinder_sampler(data['kdtree_1'], data_dict_1)
        
        return {
            'pc_0': sampled_data_0['pos'],
            'pc_1': sampled_data_1['pos'],
            'change_map': sampled_data_1['change_map'],
            'point_idx_pc0': sampled_data_0['point_idx'],
            'point_idx_pc1': sampled_data_1['point_idx'],
            'idx': idx
        }

    def _extract_center_info(self, centers_dict: Dict[str, torch.Tensor], idx: int) -> Tuple[torch.Tensor, int]:
        """Extracts center position and datapoint index from the centers dictionary.
        
        Parameters
        ----------
        centers_dict : dict
            Dictionary containing 'pos', 'idx', and 'change_map' tensors
        idx : int
            Index to extract
        """
        return centers_dict['pos'][idx], centers_dict['idx'][idx].int()
