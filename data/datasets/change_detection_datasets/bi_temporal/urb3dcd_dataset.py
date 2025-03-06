from typing import Tuple, Dict, Any
import os
import glob
import random
import numpy as np
import torch
from sklearn.neighbors import KDTree
from torch_points3d.core.data_transform import GridSampling3D, CylinderSampling
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

    def __init__(self, sample_per_epoch=100, radius=2, fix_samples=False, nameInPly="params", version="v1", *args, **kwargs):
        super(Urb3DCDDataset, self).__init__(*args, **kwargs)
        self._sample_per_epoch = sample_per_epoch
        self._radius = radius
        self.fix_samples = fix_samples
        self.nameInPly = nameInPly
        self.version = version
        self._grid_sampling = GridSampling3D(size=radius / 10.0)  # Renamed to be more generic
        self._init_annotations()

    def _init_annotations(self) -> None:
        """Initialize file paths for point clouds."""
        if self.version == "v1":
            base_dir = os.path.join(self.data_root, "IEEE_Dataset_V1", "1-Lidar05", self.SPLIT_MAP[self.split])
        else:
            raise ValueError(f"Version {self.version} is not supported yet")

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
        
        for i in range(len(self.annotations)):
            data = self._load_point_cloud_pair(i)
            low_res = self._grid_sampling(data['pc_1'])
            centres = torch.empty((low_res.shape[0], 5), dtype=torch.float)
            centres[:, :3] = low_res
            centres[:, 3] = i
            centres[:, 4] = data['change_map']
            self._centres_for_sampling.append(centres)
        
        self._centres_for_sampling = torch.cat(self._centres_for_sampling, 0)
        labels, label_counts = np.unique(self._centres_for_sampling[:, -1].numpy(), return_counts=True)
        self._label_counts = np.sqrt(label_counts.mean() / label_counts)
        self._label_counts /= np.sum(self._label_counts)
        self._labels = labels
        self.weight_classes = torch.tensor(self._label_counts, dtype=torch.float)
        
        if self.fix_samples:
            self._prepare_fixed_sampling()
    
    def _prepare_fixed_sampling(self):
        """Fixes the sampled locations for consistency across epochs."""
        np.random.seed(1)
        chosen_labels = np.random.choice(self._labels, p=self._label_counts, size=(self._sample_per_epoch, 1))
        unique_labels, label_counts = np.unique(chosen_labels, return_counts=True)
        
        self._centres_for_sampling_fixed = []
        for label, count in zip(unique_labels, label_counts):
            valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, -1] == label]
            selected_idx = np.random.randint(low=0, high=valid_centres.shape[0], size=(count,))
            self._centres_for_sampling_fixed.append(valid_centres[selected_idx])
        
        self._centres_for_sampling_fixed = torch.cat(self._centres_for_sampling_fixed, 0)
    
    def _prepare_grid_sampling(self):
        """Prepares centers for regular grid sampling."""
        self.grid_regular_centers = []
        grid_sampling = GridSampling3D(size=self._radius / 2)
        
        for i in range(len(self.annotations)):
            data = self._load_point_cloud_pair(i)
            grid_sample_centers = grid_sampling(data['pc_1'])
            centres = torch.empty((grid_sample_centers.shape[0], 4), dtype=torch.float)
            centres[:, :3] = grid_sample_centers
            centres[:, 3] = i
            self.grid_regular_centers.append(centres)
        
        self.grid_regular_centers = torch.cat(self.grid_regular_centers, 0)
    
    def _load_point_cloud_pair(self, idx: int) -> Dict[str, Any]:
        """Loads a pair of point clouds and builds KDTrees for them."""
        pc0, pc1, change_map = self.clouds_loader(idx, nameInPly=self.nameInPly)
        
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

    def clouds_loader(self, idx: int, nameInPly="params"):
        """Loads point clouds from files."""
        print("Loading " + self.annotations[idx]['pc_1_filepath'])
        pc = utils.io.load_point_cloud(self.annotations[idx]['pc_1_filepath'], nameInPly=nameInPly)
        pc1 = pc[:, :3]
        change_map = pc[:, 3].long()  # Labels should be at the 4th column 0:X 1:Y 2:Z 3:Label
        pc0 = utils.io.load_point_cloud(self.annotations[idx]['pc_0_filepath'], nameInPly=nameInPly)[:, :3]
        return pc0.type(torch.float), pc1.type(torch.float), change_map

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
                    'idx': sample['idx'], 'idx_target': sample['idx_target'], 
                    'area': sample['area']}
        except Exception as e:
            print(f"Normalization failed: {e}")
            print(f"pc_0 shape: {pc0.shape}, pc_1 shape: {pc1.shape}")
            return None

    def _get_fixed_sample(self, idx):
        """Retrieves a fixed sample without normalization."""
        while idx < self._centres_for_sampling_fixed.shape[0]:
            centre, area_sel = self._extract_centre_info(self._centres_for_sampling_fixed, idx)
            data = self._load_point_cloud_pair(area_sel)
            
            sample = self._sample_cylinder(data, centre, area_sel)

            if sample:
                return sample
            
            idx += 1
        return None

    def _get_regular_sample(self, idx):
        """Retrieves a regular sample without normalization."""
        while idx < self.grid_regular_centers.shape[0]:
            centre, area_sel = self._extract_centre_info(self.grid_regular_centers, idx)
            data = self._load_point_cloud_pair(area_sel)
            
            sample = self._sample_cylinder(data, centre, area_sel, apply_transform=True)

            if sample:
                return sample

            print('pair not correct')
            idx += 1
        return None

    def _get_random(self):
        """Randomly selects a sample without normalization."""
        chosen_label = np.random.choice(self._labels, p=self._label_counts)
        valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, 4] == chosen_label]

        if valid_centres.shape[0] == 0:
            return None

        centre_idx = int(random.random() * (valid_centres.shape[0] - 1))
        centre = valid_centres[centre_idx]
        area_sel = centre[3].int()
        data = self._load_point_cloud_pair(area_sel)
        
        return self._sample_cylinder(data, centre[:3], area_sel, apply_transform=True)

    def _sample_cylinder(self, data, centre, area_sel, apply_transform=False):
        """Applies cylindrical sampling and optional transformations."""
        cylinder_sampler = CylinderSampling(self._radius, centre, align_origin=False)

        # Sample points using the KDTrees
        idx_pc0 = cylinder_sampler.query(data['kdtree_0'], data['pc_0'])
        idx_pc1 = cylinder_sampler.query(data['kdtree_1'], data['pc_1'])

        return {
            'pc_0': data['pc_0'][idx_pc0], 
            'pc_1': data['pc_1'][idx_pc1], 
            'change_map': data['change_map'][idx_pc1],
            'idx': idx_pc0, 
            'idx_target': idx_pc1, 
            'area': area_sel
        }

    def _extract_centre_info(self, centres, idx):
        """Extracts center position and area selection from the given array."""
        centre = centres[idx, :3]
        area_sel = centres[idx, 3].int()
        return centre, area_sel
