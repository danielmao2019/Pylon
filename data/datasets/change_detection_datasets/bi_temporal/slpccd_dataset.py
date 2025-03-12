from typing import Tuple, Dict, Any, Optional, List
import os
import glob
import random
import numpy as np
import torch
from sklearn.neighbors import KDTree
from data.datasets import BaseDataset
from utils.torch_points3d import GridSampling3D
import utils


class SLPCCDDataset(BaseDataset):
    """Street-Level Point Cloud Change Detection (SLPCCD) Dataset.
    
    The SLPCCD dataset is a street-level point cloud change detection dataset
    derived from SHREC2021. It consists of pairs of point clouds from different
    time periods (2016 and 2020) with annotated changes.
    
    Each datapoint contains:
    - Two point clouds (pc_0 for 2016, pc_1 for 2020)
    - Change labels
    - Point cloud indices for cross-cloud neighborhood search
    
    The dataset supports multi-resolution processing via hierarchical sampling.
    """

    INPUT_NAMES = ['pc_0', 'pc_1']
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
        'train': 399,  # Number from the train.txt file
        'val': 96,     # Number from the val.txt file
        'test': 129    # Number from the test.txt file
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
        split_file = os.path.join(self.data_root, f"{self.split}.txt")
        
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
                    pc_0_path = os.path.join(self.data_root, parts[0])
                    pc_1_path = os.path.join(self.data_root, parts[1])
                    
                    annotations.append({
                        'pc_0_filepath': pc_0_path,
                        'pc_1_filepath': pc_1_path
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
        # Get file paths
        pc_0_filepath = self.annotations[idx]['pc_0_filepath']
        pc_1_filepath = self.annotations[idx]['pc_1_filepath']
        
        # Load point clouds
        pc_0 = utils.io.load_point_cloud(pc_0_filepath)
        pc_1 = utils.io.load_point_cloud(pc_1_filepath)
        
        # Extract positions and labels
        pc_0_xyz = pc_0[:, :3]
        pc_1_xyz = pc_1[:, :3]
        
        # The change label is typically in the 4th column (index 3)
        # If point cloud 1 has a 4th column, use it as the change map
        if pc_1.size(1) > 3:
            change_map = pc_1[:, 3].long()
        else:
            # If no change labels are provided, use dummy labels (all zeros)
            change_map = torch.zeros(pc_1_xyz.size(0), dtype=torch.long)
        
        # Store original point cloud lengths
        pc_0_raw_length = pc_0_xyz.size(0)
        pc_1_raw_length = pc_1_xyz.size(0)
        
        # Normalize point clouds (center and scale)
        pc_0_xyz = self._normalize_point_cloud(pc_0_xyz)
        pc_1_xyz = self._normalize_point_cloud(pc_1_xyz)
        
        # Subsample or pad to fixed size
        if self.random_subsample:
            pc_0_xyz = self._random_subsample_point_cloud(pc_0_xyz, self.num_points)
            # For pc_1, also subsample or pad the change_map
            if pc_1_xyz.size(0) != self.num_points:
                if pc_1_xyz.size(0) < self.num_points:
                    # Padding case
                    indices = torch.randint(0, pc_1_xyz.size(0), (self.num_points - pc_1_xyz.size(0),))
                    change_map_padding = change_map[indices]
                    change_map = torch.cat([change_map, change_map_padding], dim=0)
                else:
                    # Subsampling case
                    indices = torch.randperm(pc_1_xyz.size(0))[:self.num_points]
                    change_map = change_map[indices]
                pc_1_xyz = self._random_subsample_point_cloud(pc_1_xyz, self.num_points)
        
        # Add simple features (ones) to each point cloud
        pc_0_feat = torch.ones((pc_0_xyz.size(0), 1), dtype=torch.float32)
        pc_1_feat = torch.ones((pc_1_xyz.size(0), 1), dtype=torch.float32)
        
        # Create point cloud dictionaries
        pc_0_dict = {'pos': pc_0_xyz, 'feat': pc_0_feat}
        pc_1_dict = {'pos': pc_1_xyz, 'feat': pc_1_feat}
        
        # Compute cross-point cloud nearest neighbors
        knearst_idx_in_another_pc_0 = self._compute_cross_knn(pc_0_xyz, pc_1_xyz, self.cross_knn_size)
        knearst_idx_in_another_pc_1 = self._compute_cross_knn(pc_1_xyz, pc_0_xyz, self.cross_knn_size)
        
        # Build hierarchical representation if enabled
        inputs = {}
        if self.use_hierarchy:
            pc_0_hierarchy = self._build_hierarchy(pc_0_dict)
            pc_1_hierarchy = self._build_hierarchy(pc_1_dict)
            
            # Combine hierarchies into a single input dictionary
            inputs['pc_0'] = {
                'xyz': pc_0_hierarchy['xyz'],
                'neighbors_idx': pc_0_hierarchy['neighbors_idx'],
                'pool_idx': pc_0_hierarchy['pool_idx'],
                'unsam_idx': pc_0_hierarchy['unsam_idx'],
                'knearst_idx_in_another_pc': knearst_idx_in_another_pc_0,
                'raw_length': pc_0_raw_length
            }
            
            inputs['pc_1'] = {
                'xyz': pc_1_hierarchy['xyz'],
                'neighbors_idx': pc_1_hierarchy['neighbors_idx'],
                'pool_idx': pc_1_hierarchy['pool_idx'],
                'unsam_idx': pc_1_hierarchy['unsam_idx'],
                'knearst_idx_in_another_pc': knearst_idx_in_another_pc_1,
                'raw_length': pc_1_raw_length
            }
        else:
            # Simple non-hierarchical representation
            inputs['pc_0'] = {
                'xyz': pc_0_xyz,
                'neighbors_idx': self._compute_knn(pc_0_xyz, self.knn_size),
                'knearst_idx_in_another_pc': knearst_idx_in_another_pc_0,
                'raw_length': pc_0_raw_length
            }
            
            inputs['pc_1'] = {
                'xyz': pc_1_xyz,
                'neighbors_idx': self._compute_knn(pc_1_xyz, self.knn_size),
                'knearst_idx_in_another_pc': knearst_idx_in_another_pc_1,
                'raw_length': pc_1_raw_length
            }
        
        # Prepare labels and metadata
        labels = {'change_map': change_map}
        
        meta_info = {
            'idx': idx,
            'pc_0_filepath': pc_0_filepath,
            'pc_1_filepath': pc_1_filepath,
            'dir_name': os.path.dirname(pc_0_filepath),
            'file_name_0': os.path.basename(pc_0_filepath),
            'file_name_1': os.path.basename(pc_1_filepath)
        }
        
        return inputs, labels, meta_info
