from typing import Tuple, Dict, Any, Optional, List
import os
import numpy as np
import torch
from sklearn.neighbors import KDTree
from data.datasets import BaseDataset
from utils.point_cloud_ops.sampling import GridSampling3D
import utils


class SLPCCDDataset(BaseDataset):
    """Street-Level Point Cloud Change Detection (SLPCCD) Dataset.

    The SLPCCD dataset is a street-level point cloud change detection dataset
    derived from SHREC2021. It consists of pairs of point clouds from different
    time periods (2016 and 2020) with annotated changes.

    Each datapoint contains:
    - Two point clouds (pc_1 for 2016, pc_2 for 2020)
    - Change labels
    - Point cloud indices for cross-cloud neighborhood search
    The dataset supports multi-resolution processing via hierarchical sampling.
    """

    INPUT_NAMES = ['pc_1', 'pc_2']
    LABEL_NAMES = ['change_map']
    NUM_CLASSES = 2  # Binary classification (changed/unchanged)
    INV_OBJECT_LABEL = {
        0: "unchanged",
        1: "changed"
    }
    CLASS_LABELS = {name: i for i, name in INV_OBJECT_LABEL.items()}
    IGNORE_LABEL = -1
    SPLIT_OPTIONS = {'train', 'val', 'test'}
    DATASET_SIZE = {
        'train': 398,  # Number from the train.txt file
        'val': 95,     # Number from the val.txt file
        'test': 128    # Number from the test.txt file
    }

    def __init__(
        self,
        num_points: Optional[int] = 8192,
        random_subsample: Optional[bool] = True,
        use_hierarchy: Optional[bool] = True,
        hierarchy_levels: Optional[int] = 3,
        knn_size: Optional[int] = 16,
        cross_knn_size: Optional[int] = 16,
        *args,
        **kwargs
    ) -> None:
        """Initialize the SLPCCD dataset.
        Args:
            num_points: Number of points to sample from each point cloud.
            random_subsample: Whether to randomly subsample points.
            use_hierarchy: Whether to use hierarchical representation.
            hierarchy_levels: Number of levels in the hierarchy.
            knn_size: Number of nearest neighbors for each point.
            cross_knn_size: Number of nearest neighbors in the other point cloud.
        """
        self.num_points = num_points
        self.random_subsample = random_subsample
        self.use_hierarchy = use_hierarchy
        self.hierarchy_levels = hierarchy_levels
        self.knn_size = knn_size
        self.cross_knn_size = cross_knn_size
        # Create grid samplers for hierarchical representation
        if self.use_hierarchy:
            self.grid_samplers = []
            for i in range(hierarchy_levels):
                size = 0.1 * (2 ** i)  # Increasing grid size for each level
                self.grid_samplers.append(GridSampling3D(size=size, mode="mean"))
        super(SLPCCDDataset, self).__init__(*args, **kwargs)

    def _init_annotations(self) -> None:
        """Initialize file paths for point cloud pairs."""
        # Read the appropriate split file (train.txt, val.txt, or test.txt)
        split_file = os.path.join(self.data_root, "data", f"{self.split}.txt")

        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        # Read file paths from the split file
        annotations = []
        with open(split_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if not line.strip():
                    continue

                parts = line.strip().split()
                if len(parts) >= 2:
                    # Convert Windows-style paths to Unix-style
                    pc_1_path = os.path.join(self.data_root, "test_seg" if self.split == "test" else "train_seg", parts[0].replace('\\', '/'))
                    pc_2_path = os.path.join(self.data_root, "test_seg" if self.split == "test" else "train_seg", parts[1].replace('\\', '/'))

                    annotations.append({
                        'pc_1_filepath': pc_1_path,
                        'pc_2_filepath': pc_2_path
                    })
        self.annotations = annotations

    def _normalize_point_cloud(self, pc: torch.Tensor) -> torch.Tensor:
        """Normalize point cloud to be centered at the origin with unit scale.

        Args:
            pc: Point cloud tensor of shape (N, 3)

        Returns:
            Normalized point cloud
        """
        # Center the point cloud
        centroid = torch.mean(pc, dim=0)
        pc = pc - centroid
        # Scale to unit sphere
        max_dist = torch.max(torch.norm(pc, dim=1))
        if max_dist > 0:
            pc = pc / max_dist
        return pc

    def _random_subsample_point_cloud(self, pc: torch.Tensor, n_points: int) -> torch.Tensor:
        """Randomly subsample or pad point cloud to have exactly n_points.

        Args:
            pc: Point cloud tensor of shape (N, D)
            n_points: Target number of points

        Returns:
            Subsampled point cloud of shape (n_points, D)
        """
        if pc.shape[0] == 0:
            # Empty point cloud, return zeros
            return torch.zeros((n_points, pc.shape[1]), dtype=pc.dtype, device=pc.device)
        if pc.shape[0] < n_points:
            # Too few points, pad with duplicates
            indices = torch.randint(0, pc.shape[0], (n_points - pc.shape[0],), device=pc.device)
            padding = pc[indices]
            return torch.cat([pc, padding], dim=0)
        elif pc.shape[0] > n_points:
            # Too many points, subsample
            indices = torch.randperm(pc.shape[0], device=pc.device)[:n_points]
            return pc[indices]
        else:
            # Exactly right number of points
            return pc

    def _compute_knn(self, pc: torch.Tensor, k: int) -> torch.Tensor:
        """Compute k-nearest neighbors for each point.

        Args:
            pc: Point cloud tensor of shape (N, 3)
            k: Number of neighbors

        Returns:
            neighbors_idx: Indices of k-nearest neighbors for each point (N, k)
        """
        kdtree = KDTree(pc.cpu().numpy())
        distances, neighbors = kdtree.query(pc.cpu().numpy(), k=k + 1)  # +1 because the first neighbor is the point itself

        # Remove self as neighbor (first column)
        neighbors = neighbors[:, 1:]

        return torch.from_numpy(neighbors.astype(np.int64))

    def _compute_cross_knn(self, pc1: torch.Tensor, pc2: torch.Tensor, k: int) -> torch.Tensor:
        """Compute k-nearest neighbors in pc2 for each point in pc1.
        Args:
            pc1: First point cloud tensor of shape (N, 3)
            pc2: Second point cloud tensor of shape (M, 3)
            k: Number of neighbors
        Returns:
            neighbors_idx: Indices of k-nearest neighbors in pc2 for each point in pc1 (N, k)
        """
        kdtree = KDTree(pc2.cpu().numpy())
        distances, neighbors = kdtree.query(pc1.cpu().numpy(), k=k)
        return torch.from_numpy(neighbors.astype(np.int64))

    def _build_hierarchy(self, pc_dict: Dict[str, torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        """Build hierarchical representation of point cloud.
        Args:
            pc_dict: Dictionary containing point cloud data
                - pos: Point positions (N, 3)
                - feat: Point features (N, D)
        Returns:
            Dictionary with hierarchical representation:
                - xyz: List of point positions at each level [(N_0, 3), (N_1, 3), ...]
                - neighbors_idx: List of neighbor indices at each level [(N_0, k), (N_1, k), ...]
                - pool_idx: List of pooling indices between levels [(N_1, 1), (N_2, 1), ...]
                - unsam_idx: List of upsampling indices between levels [(N_0, 1), (N_1, 1), ...]
        """
        hierarchy = {
            'xyz': [pc_dict['pos']],
            'feat': [pc_dict['feat']],
            'neighbors_idx': [self._compute_knn(pc_dict['pos'], self.knn_size)],
            'pool_idx': [],
            'unsam_idx': []
        }
        current_data = {
            'pos': pc_dict['pos'],
            'feat': pc_dict['feat']
        }
        # Build hierarchy levels
        for i in range(self.hierarchy_levels - 1):
            # Apply grid sampling to get the next level
            sampled_data = self.grid_samplers[i](current_data)

            # Store point positions and features
            hierarchy['xyz'].append(sampled_data['pos'])
            hierarchy['feat'].append(sampled_data['feat'])

            # Compute KNN for this level
            hierarchy['neighbors_idx'].append(self._compute_knn(sampled_data['pos'], self.knn_size))

            # Store pool and upsample indices
            hierarchy['pool_idx'].append(sampled_data['grid_idx'])
            hierarchy['unsam_idx'].append(sampled_data['inverse_indices'])

            # Update current data for next level
            current_data = sampled_data

        return hierarchy

    def _load_point_cloud_files(self, idx: int) -> Dict[str, Any]:
        """Load point cloud files for a given datapoint.
        Args:
            idx: Index of the datapoint to load

        Returns:
            Dictionary containing:
            - pc_1: First point cloud tensor
            - pc_2: Second point cloud tensor
            - pc_2_seg: Segmentation point cloud tensor (if available)
            - has_seg_file: Whether a segmentation file was found
            - pc_1_filepath: File path to first point cloud
            - pc_2_filepath: File path to second point cloud
        """
        # Get file paths
        pc_1_filepath = self.annotations[idx]['pc_1_filepath']
        pc_2_filepath = self.annotations[idx]['pc_2_filepath']

        # Check if there's a segmentation file (change labels)
        pc_2_seg_filepath = pc_2_filepath.replace('.txt', '_seg.txt')
        has_seg_file = os.path.exists(pc_2_seg_filepath)

        # Load point clouds
        pc_1 = utils.io.load_point_cloud(pc_1_filepath)
        pc_2 = utils.io.load_point_cloud(pc_2_filepath)

        # Load segmentation file if available
        pc_2_seg = None
        if has_seg_file:
            pc_2_seg = utils.io.load_point_cloud(pc_2_seg_filepath)

        return {
            'pc_1': pc_1,
            'pc_2': pc_2,
            'pc_2_seg': pc_2_seg,
            'has_seg_file': has_seg_file,
            'pc_1_filepath': pc_1_filepath,
            'pc_2_filepath': pc_2_filepath
        }

    def _extract_positions_and_features(self, pc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract positions and features from point cloud tensors.

        Args:
            pc_data: Dictionary containing point cloud tensors

        Returns:
            Dictionary with extracted positions and features
        """
        # Extract positions
        pc_1_xyz = pc_data['pc_1'][:, :3]
        pc_2_xyz = pc_data['pc_2'][:, :3]

        # Extract features (RGB if available, otherwise ones)
        if pc_data['pc_1'].size(1) >= 6:  # Has RGB values
            pc_1_feat = pc_data['pc_1'][:, 3:6]  # RGB values as features
        else:
            pc_1_feat = torch.ones((pc_1_xyz.size(0), 1), dtype=torch.float32)

        if pc_data['pc_2'].size(1) >= 6:  # Has RGB values
            pc_2_feat = pc_data['pc_2'][:, 3:6]  # RGB values as features
        else:
            pc_2_feat = torch.ones((pc_2_xyz.size(0), 1), dtype=torch.float32)

        # Store original point cloud lengths
        pc_1_raw_length = pc_1_xyz.size(0)
        pc_2_raw_length = pc_2_xyz.size(0)

        return {
            'pc_1_xyz': pc_1_xyz,
            'pc_2_xyz': pc_2_xyz,
            'pc_1_feat': pc_1_feat,
            'pc_2_feat': pc_2_feat,
            'pc_1_raw_length': pc_1_raw_length,
            'pc_2_raw_length': pc_2_raw_length
        }

    def _extract_change_map(self, pc_data: Dict[str, Any], pc_2_xyz: torch.Tensor) -> torch.Tensor:
        """Extract change map labels from point cloud data.

        Args:
            pc_data: Dictionary containing point cloud and segmentation data
            pc_2_xyz: Second point cloud positions tensor

        Returns:
            Change map tensor
        """
        # Load change labels
        if pc_data['has_seg_file'] and pc_data['pc_2_seg'] is not None:
            # Load segmentation file containing change labels
            change_map = pc_data['pc_2_seg'][:, 3].long()  # 4th column of seg file contains labels
        else:
            # If no segmentation file, check if pc_2 has labels in 4th column
            if pc_data['pc_2'].size(1) > 3:
                change_map = pc_data['pc_2'][:, 3].long()
            else:
                # If no change labels are provided, use dummy labels (all zeros)
                change_map = torch.zeros(pc_2_xyz.size(0), dtype=torch.long)

        return change_map

    def _process_point_clouds(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize and subsample point clouds and related data.

        Args:
            data: Dictionary containing point cloud positions, features, and change map

        Returns:
            Dictionary with processed point clouds
        """
        # Normalize point clouds (center and scale)
        pc_1_xyz = self._normalize_point_cloud(data['pc_1_xyz'])
        pc_2_xyz = self._normalize_point_cloud(data['pc_2_xyz'])

        # Get original features and change map
        pc_1_feat = data['pc_1_feat']
        pc_2_feat = data['pc_2_feat']
        change_map = data['change_map']

        # Subsample or pad to fixed size
        if self.random_subsample:
            # Subsample first point cloud
            pc_1_xyz = self._random_subsample_point_cloud(pc_1_xyz, self.num_points)
            pc_1_feat = self._random_subsample_point_cloud(pc_1_feat, self.num_points)

            # For second point cloud, also subsample or pad the change_map
            if pc_2_xyz.size(0) != self.num_points:
                if pc_2_xyz.size(0) < self.num_points:
                    # Padding case
                    indices = torch.randint(0, pc_2_xyz.size(0), (self.num_points - pc_2_xyz.size(0),))
                    change_map_padding = change_map[indices]
                    change_map = torch.cat([change_map, change_map_padding], dim=0)
                else:
                    # Subsampling case
                    indices = torch.randperm(pc_2_xyz.size(0))[:self.num_points]
                    change_map = change_map[indices]

                # Subsample second point cloud
                pc_2_xyz = self._random_subsample_point_cloud(pc_2_xyz, self.num_points)
                pc_2_feat = self._random_subsample_point_cloud(pc_2_feat, self.num_points)

        return {
            'pc_1_xyz': pc_1_xyz,
            'pc_2_xyz': pc_2_xyz,
            'pc_1_feat': pc_1_feat,
            'pc_2_feat': pc_2_feat,
            'change_map': change_map
        }

    def _compute_neighborhood_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute nearest neighbors within and across point clouds.

        Args:
            data: Dictionary containing processed point cloud positions

        Returns:
            Dictionary with KNN information
        """
        # Create point cloud dictionaries
        pc_1_dict = {'pos': data['pc_1_xyz'], 'feat': data['pc_1_feat']}
        pc_2_dict = {'pos': data['pc_2_xyz'], 'feat': data['pc_2_feat']}

        # Compute cross-point cloud nearest neighbors
        knearst_idx_in_another_pc_1 = self._compute_cross_knn(data['pc_1_xyz'], data['pc_2_xyz'], self.cross_knn_size)
        knearst_idx_in_another_pc_2 = self._compute_cross_knn(data['pc_2_xyz'], data['pc_1_xyz'], self.cross_knn_size)

        return {
            'pc_1_dict': pc_1_dict,
            'pc_2_dict': pc_2_dict,
            'knearst_idx_in_another_pc_1': knearst_idx_in_another_pc_1,
            'knearst_idx_in_another_pc_2': knearst_idx_in_another_pc_2
        }

    def _build_input_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build the final input structure based on hierarchy setting.

        Args:
            data: Dictionary containing all processed point cloud data

        Returns:
            Dictionary with input structure
        """
        inputs = {}

        if self.use_hierarchy:
            # Build hierarchical representation for both point clouds
            pc_1_hierarchy = self._build_hierarchy(data['pc_1_dict'])
            pc_2_hierarchy = self._build_hierarchy(data['pc_2_dict'])

            # Add cross-point cloud nearest neighbors
            pc_1_hierarchy['knearst_idx_in_another_pc'] = data['knearst_idx_in_another_pc_1']
            pc_2_hierarchy['knearst_idx_in_another_pc'] = data['knearst_idx_in_another_pc_2']

            # Add raw lengths
            pc_1_hierarchy['raw_length'] = data['pc_1_raw_length']
            pc_2_hierarchy['raw_length'] = data['pc_2_raw_length']

            inputs['pc_1'] = pc_1_hierarchy
            inputs['pc_2'] = pc_2_hierarchy
        else:
            # Simple non-hierarchical representation
            inputs['pc_1'] = {
                'xyz': data['pc_1_dict']['pos'],
                'neighbors_idx': self._compute_knn(data['pc_1_dict']['pos'], self.knn_size),
                'knearst_idx_in_another_pc': data['knearst_idx_in_another_pc_1'],
                'raw_length': data['pc_1_raw_length'],
                'feat': data['pc_1_dict']['feat']
            }

            inputs['pc_2'] = {
                'xyz': data['pc_2_dict']['pos'],
                'neighbors_idx': self._compute_knn(data['pc_2_dict']['pos'], self.knn_size),
                'knearst_idx_in_another_pc': data['knearst_idx_in_another_pc_2'],
                'raw_length': data['pc_2_raw_length'],
                'feat': data['pc_2_dict']['feat']
            }

        return inputs

    def _prepare_meta_info(self, idx: int, pc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare metadata for the datapoint.

        Args:
            idx: Index of the datapoint
            pc_data: Dictionary containing file paths

        Returns:
            Dictionary with metadata
        """
        return {
            'idx': idx,
            'pc_1_filepath': pc_data['pc_1_filepath'],
            'pc_2_filepath': pc_data['pc_2_filepath'],
            'dir_name': os.path.dirname(pc_data['pc_1_filepath']),
            'file_name_1': os.path.basename(pc_data['pc_1_filepath']),
            'file_name_2': os.path.basename(pc_data['pc_2_filepath'])
        }

    def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        """Load a datapoint from the dataset.

        Args:
            idx: Index of the datapoint to load

        Returns:
            Tuple containing:
            - inputs: Dictionary of input tensors
            - labels: Dictionary of label tensors
            - meta_info: Dictionary of metadata
        """
        # Step 1: Load point cloud files
        pc_data = self._load_point_cloud_files(idx)

        # Step 2: Extract positions and features
        extracted_data = self._extract_positions_and_features(pc_data)

        # Step 3: Extract change map
        change_map = self._extract_change_map(pc_data, extracted_data['pc_2_xyz'])

        # Combine data for processing
        data_for_processing = {
            **extracted_data,
            'change_map': change_map
        }

        # Step 4: Process point clouds (normalize and subsample)
        processed_data = self._process_point_clouds(data_for_processing)

        # Combine data for neighborhood computation
        data_for_neighborhood = {
            **processed_data,
            'pc_1_raw_length': extracted_data['pc_1_raw_length'],
            'pc_2_raw_length': extracted_data['pc_2_raw_length']
        }

        # Step 5: Compute neighborhood information
        neighborhood_data = self._compute_neighborhood_info(data_for_neighborhood)

        # Combine all data for input structure
        all_data = {
            **data_for_neighborhood,
            **neighborhood_data
        }

        # Step 6: Build final input structure
        inputs = self._build_input_structure(all_data)

        # Step 7: Prepare labels and metadata
        labels = {'change_map': processed_data['change_map']}
        meta_info = self._prepare_meta_info(idx, pc_data)

        return inputs, labels, meta_info
