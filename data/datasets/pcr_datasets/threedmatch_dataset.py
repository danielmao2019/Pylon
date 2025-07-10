from typing import Any, Dict, Tuple
import os
import pickle
import numpy as np
import torch
from data.datasets.base_dataset import BaseDataset
from utils.point_cloud_ops.correspondences import get_correspondences
from utils.io.point_cloud import load_point_cloud_tensor


class ThreeDMatchDataset(BaseDataset):
    """3DMatch dataset for point cloud registration.
    
    This dataset contains RGB-D scans of real-world indoor scenes from the 3DMatch benchmark.
    It is commonly used for evaluating point cloud registration algorithms.
    
    Paper:
        3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions
        https://arxiv.org/abs/1603.08182
    """
    
    SPLIT_OPTIONS = ['train', 'val', 'test']
    DATASET_SIZE = {
        'train': 20642,
        'val': 2000, 
        'test': 1623,
    }
    INPUT_NAMES = ['src_pc', 'tgt_pc', 'correspondences']
    LABEL_NAMES = ['transform']
    SHA1SUM = None
    
    def __init__(
        self,
        matching_radius: float = 0.1,
        overlap_threshold: float = 0.3,
        **kwargs,
    ) -> None:
        """Initialize the dataset.
        
        Args:
            matching_radius: Radius for finding correspondences (default: 0.1)
            overlap_threshold: Minimum overlap ratio between point cloud pairs (default: 0.3)
            **kwargs: Additional arguments passed to BaseDataset
        """
        self.matching_radius = matching_radius
        self.overlap_threshold = overlap_threshold
        
        # Initialize base class
        super(ThreeDMatchDataset, self).__init__(**kwargs)
    
    def _init_annotations(self) -> None:
        """Initialize dataset annotations from metadata files."""
        # Metadata paths
        metadata_dir = os.path.join(self.data_root, 'metadata')
        data_dir = os.path.join(self.data_root, 'data')
        
        # Load metadata based on split (always use 3DMatch for test)
        if self.split in ['train', 'val']:
            metadata_file = os.path.join(metadata_dir, f'{self.split}.pkl')
        else:  # test split
            metadata_file = os.path.join(metadata_dir, '3DMatch.pkl')
        
        # Load metadata
        with open(metadata_file, 'rb') as f:
            metadata_list = pickle.load(f)
        
        # Filter by overlap threshold
        if self.overlap_threshold is not None:
            metadata_list = [x for x in metadata_list if x['overlap'] > self.overlap_threshold]
        
        # Convert metadata to annotations format
        self.annotations = []
        for item in metadata_list:
            annotation = {
                'src_path': os.path.join(data_dir, item['pcd0']),
                'tgt_path': os.path.join(data_dir, item['pcd1']),
                'rotation': item['rotation'],  # (3, 3) numpy array
                'translation': item['translation'],  # (3,) numpy array
                'overlap': item['overlap'],
                'scene_name': item['scene_name'],
                'frag_id0': item['frag_id0'],
                'frag_id1': item['frag_id1'],
            }
            self.annotations.append(annotation)
        
        # Validate dataset size
        expected_size = self.DATASET_SIZE[self.split] 
        actual_size = len(self.annotations)
        print(f'Dataset {self.split}: expected {expected_size}, got {actual_size} samples after filtering')
    
    def _load_point_cloud(self, file_path: str, device: torch.device) -> torch.Tensor:
        """Load point cloud tensor from file.
        
        Args:
            file_path: Path to the point cloud file
            device: Device to load the tensor on
            
        Returns:
            Point cloud tensor of shape (N, 3)
        """
        # Use the specialized tensor loader that handles device placement optimally
        return load_point_cloud_tensor(file_path, device=device)
    
    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        """Load a datapoint from the dataset.
        
        Args:
            idx: Index of the datapoint
            
        Returns:
            Tuple of (inputs, labels, meta_info)
        """
        # Get annotation
        ann = self.annotations[idx]
        
        # Load point clouds (prefer GPU if available)
        device = self.device if torch.cuda.is_available() else torch.device('cpu')
        src_points = self._load_point_cloud(ann['src_path'], device)
        tgt_points = self._load_point_cloud(ann['tgt_path'], device)
        
        # Create transformation matrix (4x4) on same device as points
        transform = torch.eye(4, dtype=torch.float32, device=device)
        transform[:3, :3] = torch.from_numpy(ann['rotation']).float().to(device)
        transform[:3, 3] = torch.from_numpy(ann['translation']).float().to(device)
        
        # Find correspondences between source and target
        correspondences = get_correspondences(
            src_points=src_points,
            tgt_points=tgt_points,
            transform=transform,
            radius=self.matching_radius,
        )
        
        # Prepare inputs
        inputs = {
            'src_pc': {
                'pos': src_points,
                'feat': torch.ones((src_points.shape[0], 1), dtype=torch.float32, device=device),
            },
            'tgt_pc': {
                'pos': tgt_points,
                'feat': torch.ones((tgt_points.shape[0], 1), dtype=torch.float32, device=device),
            },
            'correspondences': correspondences,
        }
        
        # Prepare labels
        labels = {
            'transform': transform,
        }
        
        # Prepare meta info
        meta_info = {
            'src_path': ann['src_path'],
            'tgt_path': ann['tgt_path'],
            'scene_name': ann['scene_name'],
            'overlap': ann['overlap'],
            'src_frame': ann['frag_id0'],
            'tgt_frame': ann['frag_id1'],
        }
        
        return inputs, labels, meta_info
