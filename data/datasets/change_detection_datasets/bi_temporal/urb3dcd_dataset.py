from typing import Tuple, Dict, Any, Optional, List
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

    def __init__(
        self,
        version: Optional[int] = 1,
        patched: Optional[bool] = True,
        sample_per_epoch: Optional[int] = 128,
        fix_samples: Optional[bool] = False,
        radius: Optional[float] = 100,
        *args,
        **kwargs
    ) -> None:
        if version not in self.VERSION_MAP:
            raise ValueError(f"Version {version} is not supported. Must be one of {list(self.VERSION_MAP.keys())}")

        # Check for invalid parameter combinations
        if not patched:
            if sample_per_epoch is not None and sample_per_epoch != 128:
                raise ValueError("'sample_per_epoch' should not be specified when 'patched' is False.")
            if fix_samples is not None and fix_samples != False:
                raise ValueError("'fix_samples' should not be specified when 'patched' is False.")
            if radius is not None and radius != 100:
                raise ValueError("'radius' should not be specified when 'patched' is False.")

        self._sample_per_epoch = sample_per_epoch
        self.fix_samples = fix_samples
        self._radius = radius
        self.version = version
        self.patched = patched  # Whether to use patched (sampled) point clouds or full point clouds
        self._grid_sampling = GridSampling3D(size=radius / 10.0)  # Renamed to be more generic
        super(Urb3DCDDataset, self).__init__(*args, **kwargs)

    def _init_annotations(self) -> None:
        """Initialize file paths and prepare centers for sampling.

        This method performs the following steps:
        1. Gets file paths for point cloud pairs
        2. If patched mode, prepares centers for sampling; otherwise just store file paths
        3. Calculates label statistics if needed for balanced sampling
        4. Prepares final annotations based on sampling mode (fixed, random, or grid)
        """
        # Get file paths
        version_info = self.VERSION_MAP[self.version]
        base_dir = os.path.join(self.data_root, version_info['dir'], version_info['subdir'], self.SPLIT_MAP[self.split])

        # Find all point cloud pairs using glob
        pc0_files = sorted(glob.glob(os.path.join(base_dir, "**/pointCloud0.ply"), recursive=True))
        pc1_files = sorted(glob.glob(os.path.join(base_dir, "**/pointCloud1.ply"), recursive=True))

        if len(pc0_files) != len(pc1_files):
            raise ValueError(f"Number of pointCloud0 files ({len(pc0_files)}) does not match pointCloud1 files ({len(pc1_files)})")

        if len(pc0_files) == 0:
            raise ValueError(f"No point cloud pairs found in {base_dir}")

        # Store file paths in annotations
        self.annotations = [
            {'pc_0_filepath': pc0, 'pc_1_filepath': pc1}
            for pc0, pc1 in zip(pc0_files, pc1_files)
        ]

        # For non-patched mode, just use the file paths as is
        if not self.patched:
            return

        # For patched mode, continue with center preparation
        # Get all potential centers
        all_centers = self._prepare_all_centers()

        # Calculate label statistics for balanced sampling if needed
        if self._sample_per_epoch > 0:
            labels, label_counts = torch.unique(all_centers['change_map'], return_counts=True)
            self._label_counts = torch.sqrt(label_counts.float().mean() / label_counts.float())
            self._label_counts /= self._label_counts.sum()
            self._labels = labels
            self.weight_classes = self._label_counts.clone()

        # Prepare annotations based on sampling mode
        if self._sample_per_epoch > 0:
            if self.fix_samples:
                self.annotations = self._prepare_fixed_centers(all_centers)
            else:
                self.annotations = self._prepare_random_centers(all_centers)
        else:
            self.annotations = self._prepare_grid_centers(all_centers)

    def _prepare_all_centers(self) -> Dict[str, Any]:
        """Prepare all potential centers by grid sampling each point cloud pair.

        Returns:
            A dictionary containing concatenated tensors for:
            - pos: Center positions (N, 3)
            - idx: Point cloud indices (N,)
            - change_map: Change labels (N,)
            - pc_0_filepath: List of file paths for first point clouds
            - pc_1_filepath: List of file paths for second point clouds
        """
        centers_list = []
        for idx in range(len(self.annotations)):
            # Load point cloud but skip sampling during initialization
            data = self._load_point_cloud_whole(idx)
            data_dict = {
                'pos': data['pc_1'],
                'change_map': data['change_map']
            }
            sampled_data = self._grid_sampling(data_dict)
            centers = {
                'pos': sampled_data['pos'],
                'change_map': sampled_data['change_map'],
                'idx': idx * torch.ones(len(sampled_data['pos']), dtype=torch.long),
                'pc_0_filepath': self.annotations[idx]['pc_0_filepath'],
                'pc_1_filepath': self.annotations[idx]['pc_1_filepath']
            }
            centers_list.append(centers)

        # Convert to single dictionary with concatenated tensors
        return {
            'pos': torch.cat([c['pos'] for c in centers_list], dim=0),
            'change_map': torch.cat([c['change_map'] for c in centers_list], dim=0),
            'idx': torch.cat([c['idx'] for c in centers_list], dim=0),
            'pc_0_filepath': [c['pc_0_filepath'] for c in centers_list],
            'pc_1_filepath': [c['pc_1_filepath'] for c in centers_list]
        }

    def _prepare_fixed_centers(self, all_centers: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare fixed centers with balanced sampling.

        Args:
            all_centers: Dictionary containing all potential centers and their metadata.
                Must contain 'pos', 'idx', 'change_map', 'pc_0_filepath', and 'pc_1_filepath'.

        Returns:
            List of dictionaries, each containing center information for one sample:
            - pos: Center position (3,)
            - idx: Point cloud index (scalar)
            - pc_0_filepath: File path for first point cloud
            - pc_1_filepath: File path for second point cloud
        """
        chosen_labels = random.choices(self._labels.tolist(), weights=self._label_counts.tolist(), k=self._sample_per_epoch)
        unique_labels, counts = torch.unique(torch.tensor(chosen_labels), return_counts=True)

        fixed_centers = []
        for label, count in zip(unique_labels, counts):
            mask = all_centers['change_map'] == label
            valid_pos = all_centers['pos'][mask]
            valid_idx = all_centers['idx'][mask]
            selected_indices = torch.randint(low=0, high=valid_pos.shape[0], size=(count.item(),))
            for idx in selected_indices:
                fixed_centers.append({
                    'pos': valid_pos[idx],
                    'idx': valid_idx[idx],
                    'pc_0_filepath': all_centers['pc_0_filepath'][valid_idx[idx]],
                    'pc_1_filepath': all_centers['pc_1_filepath'][valid_idx[idx]]
                })
        return fixed_centers

    def _prepare_random_centers(self, all_centers: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare random centers for balanced sampling.

        Randomly selects samples with balanced class distribution.

        Args:
            all_centers: Dictionary containing all potential centers and their metadata.
                Must contain 'pos', 'idx', 'change_map', 'pc_0_filepath', and 'pc_1_filepath'.

        Returns:
            List of dictionaries, each containing center information for one sample:
            - pos: Center position (3,)
            - idx: Point cloud index (scalar)
            - pc_0_filepath: File path for first point cloud
            - pc_1_filepath: File path for second point cloud
        """
        chosen_labels = random.choices(self._labels.tolist(), weights=self._label_counts.tolist(), k=self._sample_per_epoch)
        random_centers = []
        for label in chosen_labels:
            # Create mask for the current label
            mask = all_centers['change_map'] == label

            # Get indices where mask is True
            mask_indices = torch.nonzero(mask).squeeze(1)

            # Randomly select one index from these mask indices
            if len(mask_indices) > 0:
                random_idx = torch.randint(low=0, high=len(mask_indices), size=(1,)).item()
                selected_idx = mask_indices[random_idx].item()

                # Use the selected index to get the corresponding center data
                random_centers.append({
                    'pos': all_centers['pos'][selected_idx],
                    'idx': all_centers['idx'][selected_idx],
                    'pc_0_filepath': all_centers['pc_0_filepath'][all_centers['idx'][selected_idx]],
                    'pc_1_filepath': all_centers['pc_1_filepath'][all_centers['idx'][selected_idx]]
                })
            else:
                print(f"Warning: No centers found for label {label}")

        return random_centers

    def _prepare_grid_centers(self, all_centers: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare grid centers for systematic coverage.

        Args:
            all_centers: Dictionary containing all potential centers and their metadata.
                Must contain 'pos', 'idx', 'change_map', 'pc_0_filepath', and 'pc_1_filepath'.

        Returns:
            List of dictionaries, each containing center information for one sample:
            - pos: Center position (3,)
            - idx: Point cloud index (scalar)
            - pc_0_filepath: File path for first point cloud
            - pc_1_filepath: File path for second point cloud
        """
        grid_centers = []
        for i in range(len(all_centers['pos'])):
            grid_centers.append({
                'pos': all_centers['pos'][i],
                'idx': all_centers['idx'][i],
                'pc_0_filepath': all_centers['pc_0_filepath'][all_centers['idx'][i]],
                'pc_1_filepath': all_centers['pc_1_filepath'][all_centers['idx'][i]]
            })
        return grid_centers

    def _load_datapoint(self, idx: int, max_attempts: int = 10) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        """Load a datapoint for the parent class interface.

        Calls either _load_datapoint_patched or _load_datapoint_whole based on self.patched setting.

        Args:
            idx: Index of the datapoint to load.
            max_attempts: Maximum number of sampling attempts before giving up (only used in patched mode).

        Returns:
            The inputs, labels, and meta_info for a valid datapoint.

        Raises:
            ValueError: If no valid datapoint could be found after max_attempts.
        """
        if self.patched:
            return self._load_datapoint_patched(idx, max_attempts)
        else:
            return self._load_datapoint_whole(idx)

    def _load_datapoint_whole(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        """Load a whole point cloud datapoint without sampling.

        Args:
            idx: Index of the datapoint to load.

        Returns:
            The inputs, labels, and meta_info for the datapoint.
        """
        # Load the point cloud pair
        pc_data = self._load_point_cloud_whole(idx)

        # Create inputs dictionary
        inputs = {
            'pc_0': pc_data['pc_0'],
            'pc_1': pc_data['pc_1'],
            'kdtree_0': pc_data['kdtree_0'],
            'kdtree_1': pc_data['kdtree_1']
        }

        # Create labels dictionary
        labels = {
            'change_map': pc_data['change_map']
        }

        # Create meta_info dictionary
        meta_info = {
            'idx': idx,
            'pc_0_filepath': self.annotations[idx]['pc_0_filepath'],
            'pc_1_filepath': self.annotations[idx]['pc_1_filepath']
        }

        return inputs, labels, meta_info

    def _load_datapoint_patched(self, idx: int, max_attempts: int = 10) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        """Load a datapoint by sampling from point clouds.

        If the sampled point clouds are empty or invalid, it will try different sampling locations
        until it finds a valid sample or reaches the maximum number of attempts.

        Args:
            idx: Index of the datapoint to load.
            max_attempts: Maximum number of sampling attempts before giving up.

        Returns:
            The inputs, labels, and meta_info for a valid datapoint.

        Raises:
            ValueError: If no valid datapoint could be found after max_attempts.
        """
        original_idx = idx
        attempts = 0

        while attempts < max_attempts:
            print(f"Attempt {attempts+1}/{max_attempts}: Loading datapoint from index {idx}")

            try:
                # Load point cloud with sampling
                pc_data = self._load_point_cloud_patched(idx)

                if pc_data is None:
                    print(f"Sampling failed at index {idx}")
                    raise ValueError("Sampling failed")

                # Extract sampled data
                pc0 = pc_data['sampled_pc_0']
                pc1 = pc_data['sampled_pc_1']
                change_map = pc_data['sampled_change_map']

                # Validate the sampled point clouds
                if not self._is_valid_datapoint(pc0, pc1, change_map):
                    print(f"Invalid datapoint at index {idx}")
                    raise ValueError("Invalid datapoint")

                # Create the inputs, labels, and meta_info
                inputs = {
                    'pc_0': pc0,
                    'pc_1': pc1,
                    'kdtree_0': KDTree(np.asarray(pc0), leaf_size=10),
                    'kdtree_1': KDTree(np.asarray(pc1), leaf_size=10),
                }

                labels = {
                    'change_map': change_map
                }

                meta_info = {
                    'idx': original_idx,
                    'center_idx': self.annotations[idx].get('idx', idx),
                    'center_pos': self.annotations[idx].get('pos', None),
                    'pc_0_filepath': self.annotations[idx]['pc_0_filepath'],
                    'pc_1_filepath': self.annotations[idx]['pc_1_filepath'],
                    'attempt': attempts + 1
                }

                # Add label counts for debugging
                labels_unique, counts = torch.unique(change_map, return_counts=True)
                for label, count in zip(labels_unique, counts):
                    label_name = self.INV_OBJECT_LABEL.get(label.item(), f"unknown_{label.item()}")
                    meta_info[f'label_{label.item()}_{label_name}_count'] = count.item()

                return inputs, labels, meta_info

            except (ValueError, AssertionError) as e:
                print(f"Error processing datapoint at index {idx}: {str(e)}")
                # Try another random index
                if len(self.annotations) > 1:
                    idx = random.randint(0, len(self.annotations) - 1)
                attempts += 1
                continue

        raise ValueError(f"Could not find a valid datapoint after {max_attempts} attempts")

    def _load_point_cloud_whole(self, idx: int) -> Dict[str, Any]:
        """Load a pair of point clouds without sampling.

        Args:
            idx: Index of the point cloud pair to load.

        Returns:
            Dictionary containing:
            - pc_0: First point cloud (N, 3)
            - pc_1: Second point cloud (M, 3)
            - change_map: Change labels for second point cloud (M,)
            - kdtree_0: KDTree for first point cloud
            - kdtree_1: KDTree for second point cloud
        """
        # Assert existence of file paths
        assert 'pc_0_filepath' in self.annotations[idx], f"Missing pc_0_filepath in annotation {idx}"
        assert 'pc_1_filepath' in self.annotations[idx], f"Missing pc_1_filepath in annotation {idx}"

        files = {
            'pc_0_filepath': self.annotations[idx]['pc_0_filepath'],
            'pc_1_filepath': self.annotations[idx]['pc_1_filepath']
        }

        print("Loading " + files['pc_1_filepath'])
        nameInPly = self.VERSION_MAP[self.version]['nameInPly']

        # Load first point cloud
        pc0 = utils.io.load_point_cloud(files['pc_0_filepath'], nameInPly=nameInPly)
        assert pc0.size(1) == 4, f"{pc0.shape=}"
        pc0_xyz = pc0[:, :3]

        # Load second point cloud
        pc = utils.io.load_point_cloud(files['pc_1_filepath'], nameInPly=nameInPly)
        assert pc.size(1) == 4, f"{pc.shape=}"
        pc1_xyz = pc[:, :3]
        change_map = pc[:, 3]  # Labels should be at the 4th column 0:X 1:Y 2:Z 3:Label

        # Convert to correct types
        pc0_xyz = pc0_xyz.type(torch.float32)
        pc1_xyz = pc1_xyz.type(torch.float32)
        change_map = change_map.type(torch.int64)

        # Build KDTrees
        kdtree_0 = KDTree(np.asarray(pc0_xyz), leaf_size=10)
        kdtree_1 = KDTree(np.asarray(pc1_xyz), leaf_size=10)

        # Return data dictionary
        return {
            'pc_0': pc0_xyz,
            'pc_1': pc1_xyz,
            'change_map': change_map,
            'kdtree_0': kdtree_0,
            'kdtree_1': kdtree_1,
        }

    def _load_point_cloud_patched(self, idx: int) -> Dict[str, Any]:
        """Load a pair of point clouds and sample them based on center position.

        Args:
            idx: Index of the point cloud pair to load.

        Returns:
            Dictionary containing base point cloud data plus:
            - sampled_pc_0: Sampled first point cloud
            - sampled_pc_1: Sampled second point cloud
            - sampled_change_map: Sampled change labels

        Raises:
            AssertionError: If required fields are missing
        """
        # First load the whole point cloud
        data = self._load_point_cloud_whole(idx)

        # Assert existence of center position for sampling
        assert 'pos' in self.annotations[idx], f"Missing pos in annotation {idx}"
        assert 'idx' in self.annotations[idx], f"Missing idx in annotation {idx}"

        # Sample cylinder
        sample = self._sample_cylinder(data, self.annotations[idx]['pos'], self.annotations[idx]['idx'])

        if sample is not None:
            data['sampled_pc_0'] = sample['pc_0']
            data['sampled_pc_1'] = sample['pc_1']
            data['sampled_change_map'] = sample['change_map']

        return data

    def _is_valid_datapoint(self, pc0: torch.Tensor, pc1: torch.Tensor, change_map: torch.Tensor) -> bool:
        """Check if a datapoint is valid.

        A datapoint is valid if both point clouds and the change map are not empty.

        Args:
            pc0: First point cloud.
            pc1: Second point cloud.
            change_map: Change map labels.

        Returns:
            True if the datapoint is valid, False otherwise.
        """
        if pc0.size(0) == 0:
            print(f"Invalid datapoint: First point cloud (pc_0) is empty with {pc0.size(0)} points")
            return False

        if pc1.size(0) == 0:
            print(f"Invalid datapoint: Second point cloud (pc_1) is empty with {pc1.size(0)} points")
            return False

        if change_map.size(0) == 0:
            print(f"Invalid datapoint: Change map is empty with {change_map.size(0)} points")
            return False

        # If we have at least some minimum number of points, consider it valid
        # This threshold can be adjusted based on requirements
        min_points = 5
        if pc0.size(0) < min_points or pc1.size(0) < min_points:
            print(f"Invalid datapoint: Not enough points. pc_0: {pc0.size(0)} points, pc_1: {pc1.size(0)} points")
            return False

        return True

    def _sample_cylinder(self, data: Dict[str, Any], center: torch.Tensor, idx: int, apply_transform: bool = False) -> Dict[str, torch.Tensor]:
        """Apply cylindrical sampling and optional transformations.

        Args:
            data: Dictionary containing point clouds and attributes.
            center: Center position for cylinder sampling (3,).
            idx: Point cloud index.
            apply_transform: Whether to apply transformations.

        Returns:
            Dictionary containing:
            - pc_0: Sampled first point cloud
            - pc_1: Sampled second point cloud
            - change_map: Sampled change labels
            - point_idx_pc0: Point indices in first point cloud
            - point_idx_pc1: Point indices in second point cloud
            - idx: Point cloud index
        """
        print(f"\nSampling cylinder at center {center}, radius {self._radius}")
        print(f"Point cloud shapes: pc_0={data['pc_0'].shape}, pc_1={data['pc_1'].shape}")

        cylinder_sampler = CylinderSampling(self._radius, center, align_origin=False)

        # Create data dictionaries for both point clouds
        data_dict_0 = {'pos': data['pc_0']}
        data_dict_1 = {
            'pos': data['pc_1'],
            'change_map': data['change_map']
        }

        # Sample points using the KDTrees
        print("\nSampling pc_0:")
        sampled_data_0 = cylinder_sampler(data['kdtree_0'], data_dict_0)
        print("\nSampling pc_1:")
        sampled_data_1 = cylinder_sampler(data['kdtree_1'], data_dict_1)

        result = {
            'pc_0': sampled_data_0['pos'],
            'pc_1': sampled_data_1['pos'],
            'change_map': sampled_data_1['change_map'],
            'point_idx_pc0': sampled_data_0['point_idx'],
            'point_idx_pc1': sampled_data_1['point_idx'],
            'idx': idx
        }

        print(f"\nSampling results:")
        print(f"  pc_0 shape: {result['pc_0'].shape}")
        print(f"  pc_1 shape: {result['pc_1'].shape}")
        return result

    def _normalize(self, pc0: torch.Tensor, pc1: torch.Tensor) -> None:
        """Normalize point clouds by centering them at the origin.

        Args:
            pc0: First point cloud (N, 3)
            pc1: Second point cloud (M, 3)

        Raises:
            ValueError: If both point clouds are empty.
        """
        if pc0.shape[0] == 0 and pc1.shape[0] == 0:
            raise ValueError("Cannot normalize: both point clouds are empty")

        # If one point cloud is empty, use the other's min values
        if pc0.shape[0] == 0:
            min0 = torch.unsqueeze(pc1.min(0)[0], 0)
            min1 = min0
        elif pc1.shape[0] == 0:
            min0 = torch.unsqueeze(pc0.min(0)[0], 0)
            min1 = min0
        else:
            min0 = torch.unsqueeze(pc0.min(0)[0], 0)
            min1 = torch.unsqueeze(pc1.min(0)[0], 0)

        minG = torch.cat((min0, min1), axis=0).min(0)[0]

        if pc0.shape[0] > 0:
            pc0[:, 0] = (pc0[:, 0] - minG[0])  # x
            pc0[:, 1] = (pc0[:, 1] - minG[1])  # y
            pc0[:, 2] = (pc0[:, 2] - minG[2])  # z

        if pc1.shape[0] > 0:
            pc1[:, 0] = (pc1[:, 0] - minG[0])  # x
            pc1[:, 1] = (pc1[:, 1] - minG[1])  # y
            pc1[:, 2] = (pc1[:, 2] - minG[2])  # z
