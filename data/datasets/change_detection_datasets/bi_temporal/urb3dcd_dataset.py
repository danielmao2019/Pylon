from typing import Tuple, Dict, Any
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

    def __init__(self, sample_per_epoch=100, radius=2, fix_samples=False, version=1, *args, **kwargs):
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

    def _prepare_centers(self):
        """Prepares centers based on whether random sampling or grid sampling is used."""
        if self._sample_per_epoch > 0:
            self._prepare_random_sampling()
        else:
            self._prepare_grid_sampling()

    def _prepare_random_sampling(self):
        """Prepares centers for random sampling."""
        self._centres_for_sampling = []
        
        for idx in range(len(self.annotations)):
            data = self._load_point_cloud_pair(idx)
            
            # Create data dictionary for grid sampling
            data_dict = {
                'pos': data['pc_1'],
                'change_map': data['change_map']
            }
            
            # Apply grid sampling
            sampled_data = self._grid_sampling(data_dict)
            
            # Create a single dictionary for all sampled points
            centres = {
                'pos': sampled_data['pos'],
                'idx': idx * torch.ones(len(sampled_data['pos']), dtype=torch.long),
                'change_map': sampled_data['change_map']
            }
            self._centres_for_sampling.append(centres)
        
        # Convert to tensors for efficient indexing
        all_pos = torch.cat([c['pos'] for c in self._centres_for_sampling], dim=0)
        all_idx = torch.cat([c['idx'] for c in self._centres_for_sampling], dim=0)
        all_change_map = torch.cat([c['change_map'] for c in self._centres_for_sampling], dim=0)
        
        # Store as a single dictionary
        self._centres_for_sampling = {
            'pos': all_pos,
            'idx': all_idx,
            'change_map': all_change_map
        }
        
        # Calculate label statistics
        labels, label_counts = np.unique(all_change_map.numpy(), return_counts=True)
        self._label_counts = np.sqrt(label_counts.mean() / label_counts)
        self._label_counts /= np.sum(self._label_counts)
        self._labels = labels
        self.weight_classes = torch.tensor(self._label_counts, dtype=torch.float32)
        
        if self.fix_samples:
            self._prepare_fixed_sampling()
    
    def _prepare_fixed_sampling(self):
        """Fixes the sampled locations for consistency across epochs."""
        np.random.seed(1)
        chosen_labels = np.random.choice(self._labels, p=self._label_counts, size=(self._sample_per_epoch, 1))
        unique_labels, label_counts = np.unique(chosen_labels, return_counts=True)
        
        self._centres_for_sampling_fixed = []
        for label, count in zip(unique_labels, label_counts):
            valid_centres = self._centres_for_sampling['pos'][self._centres_for_sampling['change_map'] == label]
            selected_idx = np.random.randint(low=0, high=valid_centres.shape[0], size=(count,))
            self._centres_for_sampling_fixed.append(valid_centres[selected_idx])
        
        self._centres_for_sampling_fixed = torch.cat(self._centres_for_sampling_fixed, 0)
    
    def _prepare_grid_sampling(self):
        """Prepares centers for regular grid sampling."""
        self.grid_regular_centers = []
        grid_sampling = GridSampling3D(size=self._radius / 2)
        
        for idx in range(len(self.annotations)):
            data = self._load_point_cloud_pair(idx)
            grid_sample_centers = grid_sampling(data['pc_1'])
            centres = torch.empty((grid_sample_centers.shape[0], 4), dtype=torch.float32)
            centres[:, :3] = grid_sample_centers
            centres[:, 3] = idx  # Store datapoint index
            self.grid_regular_centers.append(centres)
        
        self.grid_regular_centers = torch.cat(self.grid_regular_centers, 0)
    
    def _load_point_cloud_pair(self, idx: int) -> Dict[str, Any]:
        """Loads a pair of point clouds and builds KDTrees for them."""
        pc0, pc1, change_map = self.clouds_loader(idx)
        
        data = {
            'pc_0': pc0,
            'pc_1': pc1,
            'change_map': change_map,
            'kdtree_0': KDTree(np.asarray(pc0), leaf_size=10),
            'kdtree_1': KDTree(np.asarray(pc1), leaf_size=10),
        }
        return data

    def __len__(self):
        return self._sample_per_epoch if self._sample_per_epoch > 0 else self.grid_regular_centers.shape[0]

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        pc0, pc1, change_map = self.get(idx)
        inputs = {
            'pc_0': pc0,
            'pc_1': pc1,
            'kdtree_0': self._loaded_data[idx]['kdtree_0'],
            'kdtree_1': self._loaded_data[idx]['kdtree_1'],
        }
        labels = {
            'change_map': change_map,
        }
        meta_info = self.annotations[idx].copy()
        return inputs, labels, meta_info

    def clouds_loader(self, idx: int):
        """Loads point clouds from files."""
        print("Loading " + self.annotations[idx]['pc_1_filepath'])
        nameInPly = self.VERSION_MAP[self.version]['nameInPly']
        pc0 = utils.io.load_point_cloud(self.annotations[idx]['pc_0_filepath'], nameInPly=nameInPly)
        assert pc0.size(1) == 3, f"{pc0.shape=}"
        # pc0 = pc0[:, :3]
        pc = utils.io.load_point_cloud(self.annotations[idx]['pc_1_filepath'], nameInPly=nameInPly)
        assert pc.size(1) == 4, f"{pc.shape=}"
        pc1 = pc[:, :3]
        change_map = pc[:, 3]  # Labels should be at the 4th column 0:X 1:Y 2:Z 3:Label
        return pc0.type(torch.float32), pc1.type(torch.float32), change_map.type(torch.int64)

    def _normalize(self, pc0, pc1):
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

    def get(self, idx):
        """Main method to get a sample and handle normalization in one place."""
        if self._sample_per_epoch > 0:
            sample = self._get_fixed_sample(idx) if self.fix_samples else self._get_random()
        else:
            sample = self._get_regular_sample(idx)
            
        if sample is None:
            return None
        
        pc0, pc1, change_map = sample['pc_0'], sample['pc_1'], sample['change_map']
            
        try:
            pc0, pc1 = self._normalize(pc0, pc1)
            return {'pc_0': pc0, 'pc_1': pc1, 'change_map': change_map, 
                    'point_idx_pc0': sample['point_idx_pc0'], 
                    'point_idx_pc1': sample['point_idx_pc1'], 
                    'idx': sample['idx']}
        except Exception as e:
            print(f"Normalization failed: {e}")
            print(f"pc_0 shape: {pc0.shape}, pc_1 shape: {pc1.shape}")
            return None

    def _get_fixed_sample(self, idx):
        """Retrieves a fixed sample without normalization."""
        while idx < self._centres_for_sampling_fixed.shape[0]:
            centre, sample_idx = self._extract_centre_info(self._centres_for_sampling_fixed, idx)
            data = self._load_point_cloud_pair(sample_idx)
            
            sample = self._sample_cylinder(data, centre, sample_idx)

            if sample:
                return sample
            
            idx += 1
        return None

    def _get_regular_sample(self, idx):
        """Retrieves a regular sample without normalization."""
        while idx < self.grid_regular_centers.shape[0]:
            centre, sample_idx = self._extract_centre_info(self.grid_regular_centers, idx)
            data = self._load_point_cloud_pair(sample_idx)
            
            sample = self._sample_cylinder(data, centre, sample_idx, apply_transform=True)

            if sample:
                return sample

            print('pair not correct')
            idx += 1
        return None

    def _get_random(self):
        """Randomly selects a sample without normalization."""
        chosen_label = np.random.choice(self._labels, p=self._label_counts)
        valid_centres = self._centres_for_sampling['pos'][self._centres_for_sampling['change_map'] == chosen_label]

        if valid_centres.shape[0] == 0:
            return None

        centre_idx = int(random.random() * (valid_centres.shape[0] - 1))
        centre = valid_centres[centre_idx]
        sample_idx = centre[3].int()
        data = self._load_point_cloud_pair(sample_idx)
        
        return self._sample_cylinder(data, centre[:3], sample_idx, apply_transform=True)

    def _sample_cylinder(self, data, centre, idx, apply_transform=False):
        """Applies cylindrical sampling and optional transformations."""
        cylinder_sampler = CylinderSampling(self._radius, centre, align_origin=False)
        
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

    def _extract_centre_info(self, centres, idx):
        """Extracts center position and datapoint index from the given array."""
        centre = centres[idx, :3]
        sample_idx = centres[idx, 3].int()
        return centre, sample_idx
