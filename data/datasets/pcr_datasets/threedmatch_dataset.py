from typing import Any, Dict, List, Tuple, Optional
import os
import pickle
import numpy as np
import torch
from data.datasets.base_dataset import BaseDataset
from utils.point_cloud_ops.correspondences import get_correspondences


class ThreeDMatchDataset(BaseDataset):
    """3DMatch dataset for point cloud registration.
    
    This dataset contains RGB-D scans of real-world indoor scenes from the 3DMatch benchmark.
    It is commonly used for evaluating point cloud registration algorithms.
    
    Paper:
        3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions
        https://arxiv.org/abs/1603.08182
    """
    
    SPLIT_OPTIONS = ['train', 'val', 'test']
    DATASET_SIZE = None  # Will be set dynamically in _init_annotations
    INPUT_NAMES = ['src_pc', 'tgt_pc', 'correspondences']
    LABEL_NAMES = ['transform']
    SHA1SUM = None
    
    def __init__(
        self,
        num_points: int = 5000,
        matching_radius: float = 0.1,
        overlap_threshold: float = 0.3,
        benchmark_mode: str = '3DMatch',  # '3DMatch' or '3DLoMatch'
        **kwargs,
    ) -> None:
        """Initialize the dataset.
        
        Args:
            num_points: Number of points to sample from each point cloud (default: 5000)
            matching_radius: Radius for finding correspondences (default: 0.1)
            overlap_threshold: Minimum overlap ratio between point cloud pairs (default: 0.3)
            benchmark_mode: Which benchmark to use ('3DMatch' or '3DLoMatch') (default: '3DMatch')
            **kwargs: Additional arguments passed to BaseDataset
        """
        assert benchmark_mode in ['3DMatch', '3DLoMatch'], f"Invalid benchmark_mode: {benchmark_mode}"
        
        self.num_points = num_points
        self.matching_radius = matching_radius
        self.overlap_threshold = overlap_threshold
        self.benchmark_mode = benchmark_mode
        
        # Initialize base class
        super(ThreeDMatchDataset, self).__init__(**kwargs)
    
    def _init_annotations(self) -> None:
        """Initialize dataset annotations from metadata files."""
        # Metadata paths
        metadata_dir = os.path.join(self.data_root, 'metadata')
        data_dir = os.path.join(self.data_root, 'data')
        
        # Load metadata based on split
        if self.split in ['train', 'val']:
            metadata_file = os.path.join(metadata_dir, f'{self.split}.pkl')
        else:  # test split
            metadata_file = os.path.join(metadata_dir, f'{self.benchmark_mode}.pkl')
        
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
        
        # Update dataset size
        if not hasattr(self, 'DATASET_SIZE') or self.DATASET_SIZE is None:
            self.DATASET_SIZE = {}
        self.DATASET_SIZE[self.split] = len(self.annotations)
    
    def _load_point_cloud(self, file_path: str, generator: torch.Generator) -> torch.Tensor:
        """Load point cloud and sample points if needed.
        
        Args:
            file_path: Path to the point cloud file
            generator: Random generator for deterministic sampling
            
        Returns:
            Point cloud tensor of shape (N, 3)
        """
        # Load point cloud (stored as PyTorch tensor)
        points = torch.load(file_path, map_location='cpu')
        
        # Ensure it's a tensor
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points).float()
        
        # Ensure it's float32
        if points.dtype != torch.float32:
            points = points.float()
        
        # Sample points if necessary
        if self.num_points is not None and points.shape[0] > self.num_points:
            # Use deterministic sampling with the provided generator
            indices = torch.randperm(points.shape[0], generator=generator)[:self.num_points]
            points = points[indices]
        
        return points
    
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
        
        # Create generator for deterministic randomness
        generator = torch.Generator()
        generator.manual_seed((self.base_seed or 0) + idx)
        
        # Load point clouds
        src_points = self._load_point_cloud(ann['src_path'], generator)
        tgt_points = self._load_point_cloud(ann['tgt_path'], generator)
        
        # Create transformation matrix (4x4)
        transform = torch.eye(4, dtype=torch.float32)
        transform[:3, :3] = torch.from_numpy(ann['rotation']).float()
        transform[:3, 3] = torch.from_numpy(ann['translation']).float()
        
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
                'feat': torch.ones((src_points.shape[0], 1), dtype=torch.float32),
            },
            'tgt_pc': {
                'pos': tgt_points,
                'feat': torch.ones((tgt_points.shape[0], 1), dtype=torch.float32),
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