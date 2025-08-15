from typing import Tuple, Dict, Any, Optional, List
import os
import glob
import random
import numpy as np
import torch
from sklearn.neighbors import KDTree
from utils.point_cloud_ops.sampling import GridSampling3D, CylinderSampling
from data.datasets.change_detection_datasets.base_3d_cd_dataset import Base3DCDDataset
import utils


class Urb3DCDDataset(Base3DCDDataset):
    __doc__ = r"""
    URB3DCD dataset for 3D point cloud change detection with multiple change types.

    For detailed documentation, see: docs/datasets/change_detection/bi_temporal/urb3dcd.md
    """

    INPUT_NAMES = ['pc_1', 'pc_2', 'kdtree_1', 'kdtree_2']  # Override base class with extended inputs
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
    SPLIT_OPTIONS = ['train', 'val', 'test']
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
        radius: Optional[float] = 50,
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
            if radius is not None and radius != 20:
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
        pc1_files = sorted(glob.glob(os.path.join(base_dir, "**/pointCloud0.ply"), recursive=True))
        pc2_files = sorted(glob.glob(os.path.join(base_dir, "**/pointCloud1.ply"), recursive=True))

        if len(pc1_files) != len(pc2_files):
            raise ValueError(f"Number of pointCloud1 files ({len(pc1_files)}) does not match pointCloud2 files ({len(pc2_files)})")

        # Store file paths in annotations
        self.annotations = [
            {'pc_1_filepath': pc1, 'pc_2_filepath': pc2}
            for pc1, pc2 in zip(pc1_files, pc2_files)
        ]

        if len(self.annotations) == 0:
            raise ValueError(f"No point cloud pairs found in {base_dir}")

        self.annotations = self.annotations[:1]

        if len(self.annotations) == 0:
            raise ValueError(f"No point cloud pairs found in {base_dir}")

        self.annotations = self.annotations[:1]

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
            - pc_1_filepath: List of file paths for first point clouds
            - pc_2_filepath: List of file paths for second point clouds
        """
        centers_list = []
        for idx in range(len(self.annotations)):
            # Load point cloud but skip sampling during initialization
            data = self._load_point_cloud_whole(idx)
            data_dict = {
                'pos': data['pc_2']['pos'],
                'change_map': data['change_map']
            }
            sampled_data = self._grid_sampling(data_dict)
            centers = {
                'pos': sampled_data['pos'],
                'change_map': sampled_data['change_map'],
                'idx': idx * torch.ones(len(sampled_data['pos']), dtype=torch.long),
                'pc_1_filepath': self.annotations[idx]['pc_1_filepath'],
                'pc_2_filepath': self.annotations[idx]['pc_2_filepath']
            }
            centers_list.append(centers)

        # Convert to single dictionary with concatenated tensors
        return {
            'pos': torch.cat([c['pos'] for c in centers_list], dim=0),
            'change_map': torch.cat([c['change_map'] for c in centers_list], dim=0),
            'idx': torch.cat([c['idx'] for c in centers_list], dim=0),
            'pc_1_filepath': [c['pc_1_filepath'] for c in centers_list],
            'pc_2_filepath': [c['pc_2_filepath'] for c in centers_list]
        }

    def _prepare_fixed_centers(self, all_centers: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare fixed centers with balanced sampling.

        Args:
            all_centers: Dictionary containing all potential centers and their metadata.
                Must contain 'pos', 'idx', 'change_map', 'pc_1_filepath', and 'pc_2_filepath'.

        Returns:
            List of dictionaries, each containing center information for one sample:
            - pos: Center position (3,)
            - idx: Point cloud index (scalar)
            - pc_1_filepath: File path for first point cloud
            - pc_2_filepath: File path for second point cloud
        """
        chosen_labels = random.choices(self._labels.tolist(), weights=self._label_counts.tolist(), k=self._sample_per_epoch)
        unique_labels, counts = torch.unique(torch.tensor(chosen_labels, device=all_centers['change_map'].device), return_counts=True)

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
                    'pc_1_filepath': all_centers['pc_1_filepath'][valid_idx[idx]],
                    'pc_2_filepath': all_centers['pc_2_filepath'][valid_idx[idx]]
                })
        return fixed_centers

    def _prepare_random_centers(self, all_centers: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare random centers for balanced sampling.

        Randomly selects samples with balanced class distribution.

        Args:
            all_centers: Dictionary containing all potential centers and their metadata.
                Must contain 'pos', 'idx', 'change_map', 'pc_1_filepath', and 'pc_2_filepath'.

        Returns:
            List of dictionaries, each containing center information for one sample:
            - pos: Center position (3,)
            - idx: Point cloud index (scalar)
            - pc_1_filepath: File path for first point cloud
            - pc_2_filepath: File path for second point cloud
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
                    'pc_1_filepath': all_centers['pc_1_filepath'][all_centers['idx'][selected_idx]],
                    'pc_2_filepath': all_centers['pc_2_filepath'][all_centers['idx'][selected_idx]]
                })
            else:
                print(f"Warning: No centers found for label {label}")

        return random_centers

    def _prepare_grid_centers(self, all_centers: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare grid centers for systematic coverage.

        Args:
            all_centers: Dictionary containing all potential centers and their metadata.
                Must contain 'pos', 'idx', 'change_map', 'pc_1_filepath', and 'pc_2_filepath'.

        Returns:
            List of dictionaries, each containing center information for one sample:
            - pos: Center position (3,)
            - idx: Point cloud index (scalar)
            - pc_1_filepath: File path for first point cloud
            - pc_2_filepath: File path for second point cloud
        """
        grid_centers = []
        for i in range(len(all_centers['pos'])):
            grid_centers.append({
                'pos': all_centers['pos'][i],
                'idx': all_centers['idx'][i],
                'pc_1_filepath': all_centers['pc_1_filepath'][all_centers['idx'][i]],
                'pc_2_filepath': all_centers['pc_2_filepath'][all_centers['idx'][i]]
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
        """Load a whole point cloud pair without sampling.

        Args:
            idx: Index of the point cloud pair to load.

        Returns:
            Tuple of (inputs, labels, meta_info) dictionaries.
        """
        # Load point cloud data
        data = self._load_point_cloud_whole(idx)

        # Create inputs dictionary - without KDTrees since they're only for data loading
        inputs = {
            'pc_1': data['pc_1'],
            'pc_2': data['pc_2']
        }

        # Create labels dictionary
        labels = {
            'change_map': data['change_map']
        }

        # Create meta_info dictionary
        meta_info = {
            'pc_1_filepath': data['pc_1_filepath'],
            'pc_2_filepath': data['pc_2_filepath']
        }

        return inputs, labels, meta_info

    def _load_datapoint_patched(self, idx: int, max_attempts: int = 10) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        """Load a point cloud patch from a pair of point clouds.

        Args:
            idx: Index of the point cloud pair to load.
            max_attempts: Maximum number of attempts to load a valid datapoint.

        Returns:
            Tuple of (inputs, labels, meta_info) where:
            - inputs: Dictionary with point cloud data
            - labels: Dictionary with change map
            - meta_info: Dictionary with metadata
        """
        attempts = 0
        while attempts < max_attempts:
            # Load the full point cloud data
            data = self._load_point_cloud_patched(idx)

            # Extract needed components
            pc1 = data['pc_1']
            pc2 = data['pc_2']
            change_map = data['change_map']

            # Check if the datapoint is valid
            if self._is_valid_datapoint(pc1, pc2, change_map):
                # Create inputs dictionary - without KDTrees since they're only for data loading
                inputs = {
                    'pc_1': pc1,
                    'pc_2': pc2
                }

                # Create labels dictionary
                labels = {
                    'change_map': change_map  # Use 'change_map' consistently
                }

                # Create meta info dictionary with all metadata
                meta_info = {
                    'center_idx': self.annotations[idx].get('idx', idx),
                    'center_pos': self.annotations[idx].get('pos', None),
                    'point_idx_pc1': data['point_idx_pc1'],
                    'point_idx_pc2': data['point_idx_pc2'],
                    'pc_1_filepath': self.annotations[idx]['pc_1_filepath'],
                    'pc_2_filepath': self.annotations[idx]['pc_2_filepath'],
                    'attempts': attempts,
                }

                return inputs, labels, meta_info
            else:
                print(f"Attempt {attempts + 1}/{max_attempts}: Invalid datapoint, retrying...")

            # Try another random index
            if idx != self.annotations[idx].get('idx', -1):
                # If this is a fixed sample, try another one
                idx = random.randint(0, len(self.annotations) - 1)
            attempts += 1

            print(f"Attempt {attempts}/{max_attempts}: Loading datapoint from index {idx}")
        # If we reach here, we exceeded the max attempts
        raise ValueError(f"Failed to load a valid datapoint after {max_attempts} attempts")

    def _load_point_cloud_whole(self, idx: int) -> Dict[str, Any]:
        """Load a pair of point clouds without sampling.

        Args:
            idx: Index of the point cloud pair to load.

        Returns:
            Dictionary containing:
            - pc_1: Dictionary with 'pos' and 'feat' for first point cloud
            - pc_2: Dictionary with 'pos' and 'feat' for second point cloud
            - change_map: Change labels for second point cloud (M,)
            - kdtree_1: KDTree for first point cloud
            - kdtree_2: KDTree for second point cloud
        """
        # Assert existence of file paths
        assert 'pc_1_filepath' in self.annotations[idx], f"Missing pc_1_filepath in annotation {idx}"
        assert 'pc_2_filepath' in self.annotations[idx], f"Missing pc_2_filepath in annotation {idx}"

        files = {
            'pc_1_filepath': self.annotations[idx]['pc_1_filepath'],
            'pc_2_filepath': self.annotations[idx]['pc_2_filepath']
        }

        print("Loading " + files['pc_2_filepath'])
        nameInPly = self.VERSION_MAP[self.version]['nameInPly']

        # Load first point cloud (only has XYZ coordinates) with float64 precision
        pc1_data = utils.io.load_point_cloud(files['pc_1_filepath'], nameInPly=nameInPly, name_feat="label_ch", dtype=torch.float64)
        pc1_xyz = pc1_data['pos']  # Extract position from dictionary
        # Add ones feature
        pc1_features = torch.ones((pc1_xyz.size(0), 1), dtype=pc1_xyz.dtype)  # [N, 1]

        # Load second point cloud (XYZ coordinates + label) with float64 precision
        pc2_data = utils.io.load_point_cloud(files['pc_2_filepath'], nameInPly=nameInPly, name_feat="label_ch", dtype=torch.float64)
        pc2_xyz = pc2_data['pos']  # Extract position from dictionary
        # Add ones feature
        pc2_features = torch.ones((pc2_xyz.size(0), 1), dtype=pc2_xyz.dtype)  # [N, 1]

        # Extract change map from features - this is mandatory, let it fail if not present
        change_map = pc2_data['feat'].squeeze()  # Labels from the loaded features

        # Convert to correct types but keep on original device
        pc1_xyz = pc1_xyz.type(torch.float32)
        pc2_xyz = pc2_xyz.type(torch.float32)
        change_map = change_map.type(torch.int64)

        # Normalize point clouds
        self._normalize(pc1_xyz, pc2_xyz)

        # Build KDTrees (move to CPU only for KDTree creation)
        kdtree_1 = KDTree(np.asarray(pc1_xyz.cpu()), leaf_size=10)
        kdtree_2 = KDTree(np.asarray(pc2_xyz.cpu()), leaf_size=10)

        # Create point indices for metadata - full point cloud so indices are just the range
        point_idx_pc1 = torch.arange(pc1_xyz.shape[0], dtype=torch.long)
        point_idx_pc2 = torch.arange(pc2_xyz.shape[0], dtype=torch.long)

        # Return data dictionary
        return {
            'pc_1': {
                'pos': pc1_xyz,
                'feat': pc1_features
            },
            'pc_2': {
                'pos': pc2_xyz,
                'feat': pc2_features
            },
            'change_map': change_map,
            'kdtree_1': kdtree_1,
            'kdtree_2': kdtree_2,
            'point_idx_pc1': point_idx_pc1,
            'point_idx_pc2': point_idx_pc2,
            'pc_1_filepath': files['pc_1_filepath'],
            'pc_2_filepath': files['pc_2_filepath'],
            'idx': idx
        }

    def _load_point_cloud_patched(self, idx: int) -> Dict[str, Any]:
        """Load a pair of point clouds and sample them based on center position.

        Args:
            idx: Index of the point cloud pair to load.

        Returns:
            Dictionary containing:
            - pc_1: Dictionary with 'pos' and 'feat' for sampled first point cloud
            - pc_2: Dictionary with 'pos' and 'feat' for sampled second point cloud
            - change_map: Sampled change labels
            - point_idx_pc1: Indices of sampled points in first point cloud
            - point_idx_pc2: Indices of sampled points in second point cloud
            - idx: Point cloud index
        """
        # First load the whole point cloud
        data = self._load_point_cloud_whole(idx)

        # Assert existence of center position for sampling
        assert 'pos' in self.annotations[idx], f"Missing pos in annotation {idx}"
        assert 'idx' in self.annotations[idx], f"Missing idx in annotation {idx}"

        # Sample a cylinder from the point cloud
        sampled_data = self._sample_cylinder(data, self.annotations[idx]['pos'], self.annotations[idx]['idx'])

        return sampled_data

    def _is_valid_datapoint(self, pc1: Dict[str, torch.Tensor], pc2: Dict[str, torch.Tensor], change_map: torch.Tensor) -> bool:
        """Check if a datapoint is valid.

        A datapoint is valid if it has at least 1 point and contains both changed and unchanged points.

        Args:
            pc1: First point cloud dictionary with keys 'pos' and 'feat'
            pc2: Second point cloud dictionary with keys 'pos' and 'feat'
            change_map: Tensor of shape (N,) containing class labels from 0 to NUM_CLASSES-1

        Returns:
            True if the datapoint is valid, False otherwise
        """
        # Check if point clouds are not empty
        if pc1['pos'].size(0) == 0 or pc2['pos'].size(0) == 0:
            print(f"Invalid datapoint: Point clouds are empty.")
            return False

        if change_map.size(0) == 0:
            print(f"Invalid datapoint: Change map is empty with {change_map.size(0)} points")
            return False

        # If we have at least some minimum number of points, consider it valid
        # This threshold can be adjusted based on requirements
        min_points = 5
        if pc1['pos'].size(0) < min_points or pc2['pos'].size(0) < min_points:
            print(f"Invalid datapoint: Not enough points. pc_1: {pc1['pos'].size(0)} points, pc_2: {pc2['pos'].size(0)} points")
            return False

        return True

    def _sample_cylinder(self, data: Dict[str, Any], center: torch.Tensor, idx: int) -> Dict[str, torch.Tensor]:
        """Apply cylindrical sampling and optional transformations.

        Args:
            data: Dictionary containing point clouds and attributes.
            center: Center position for cylinder sampling (3,).
            idx: Point cloud index.

        Returns:
            Dictionary containing:
            - pc_1: Sampled first point cloud
            - pc_2: Sampled second point cloud
            - change_map: Sampled change labels
            - point_idx_pc1: Point indices in first point cloud
            - point_idx_pc2: Point indices in second point cloud
            - idx: Point cloud index
        """
        print(f"\nSampling cylinder at center {center}, radius {self._radius}")
        print(f"Point cloud shapes: pc_1={data['pc_1']['pos'].shape}, pc_2={data['pc_2']['pos'].shape}")

        assert center.shape == (3,)
        cylinder_sampler = CylinderSampling(self._radius, center, align_origin=False)

        # Separate XYZ coordinates from features
        pc1_xyz = data['pc_1']['pos']
        pc1_features = data['pc_1']['feat']
        pc2_xyz = data['pc_2']['pos']
        pc2_features = data['pc_2']['feat']

        # Create data dictionaries for both point clouds (using only XYZ for sampling)
        data_dict_1 = {'pos': pc1_xyz}
        data_dict_2 = {'pos': pc2_xyz, 'change_map': data['change_map']}

        # Apply cylinder sampling
        sampled_data_1 = cylinder_sampler(data['kdtree_1'], data_dict_1)
        sampled_data_2 = cylinder_sampler(data['kdtree_2'], data_dict_2)

        # Return sampled data as dictionaries with separate pos and feat
        return {
            'pc_1': {
                'pos': sampled_data_1['pos'],
                'feat': pc1_features[sampled_data_1['point_idx']],
            },
            'pc_2': {
                'pos': sampled_data_2['pos'],
                'feat': pc2_features[sampled_data_2['point_idx']],
            },
            'change_map': sampled_data_2['change_map'],
            'point_idx_pc1': sampled_data_1['point_idx'],
            'point_idx_pc2': sampled_data_2['point_idx'],
            'idx': idx
        }

    def _normalize(self, pc1: torch.Tensor, pc2: torch.Tensor) -> None:
        """Normalize point clouds by centering them at their mean.

        Args:
            pc1: First point cloud (N, 3)
            pc2: Second point cloud (M, 3)

        Raises:
            ValueError: If both point clouds are empty.
        """
        if pc1.shape[0] == 0 and pc2.shape[0] == 0:
            raise ValueError("Cannot normalize: both point clouds are empty")

        # If one point cloud is empty, use the other's mean values
        if pc1.shape[0] == 0:
            mean1 = pc2.mean(0, keepdim=True)
            mean2 = mean1
        elif pc2.shape[0] == 0:
            mean1 = pc1.mean(0, keepdim=True)
            mean2 = mean1
        else:
            mean1 = pc1.mean(0, keepdim=True)
            mean2 = pc2.mean(0, keepdim=True)

        if pc1.shape[0] > 0:
            pc1.sub_(mean1)  # Center at mean

        if pc2.shape[0] > 0:
            pc2.sub_(mean2)  # Center at mean

    def _get_cache_version_dict(self) -> Dict[str, Any]:
        """Return parameters that affect dataset content for cache versioning."""
        version_dict = super()._get_cache_version_dict()
        version_dict.update({
            'version': self.version,
            'patched': self.patched,
            'sample_per_epoch': self._sample_per_epoch,
            'fix_samples': self.fix_samples,
            'radius': self._radius,
        })
        return version_dict
