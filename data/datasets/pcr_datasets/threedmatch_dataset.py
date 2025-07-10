from typing import Any, Dict, Tuple
import os
import hashlib
import pickle
import numpy as np
import torch
from data.datasets.base_dataset import BaseDataset
from utils.point_cloud_ops.correspondences import get_correspondences
from utils.io.point_cloud import load_point_cloud


class ThreeDMatchDataset(BaseDataset):
    """3DMatch dataset for point cloud registration.
    
    This dataset contains RGB-D scans of real-world indoor scenes from the 3DMatch benchmark.
    It is commonly used for evaluating point cloud registration algorithms.
    
    Paper:
        3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions
        https://arxiv.org/abs/1603.08182
    """
    
    SPLIT_OPTIONS = ['train', 'val', 'test']
    INPUT_NAMES = ['src_pc', 'tgt_pc', 'correspondences']
    LABEL_NAMES = ['transform']
    SHA1SUM = None
    DATASET_SIZE = {
        'train': 9284,
        'val': 678,
        'test': 794,
    }
    
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
        
        # Load metadata based on split (always use 3DMatch for test)
        if self.split in ['train', 'val']:
            metadata_file = os.path.join(metadata_dir, f'{self.split}.pkl')
        else:  # test split
            metadata_file = os.path.join(metadata_dir, '3DMatch.pkl')
        
        # Assert metadata file exists
        assert os.path.exists(metadata_file), f"Metadata file not found: {metadata_file}"
        
        # Load metadata
        with open(metadata_file, 'rb') as f:
            metadata_dict = pickle.load(f)
        
        # OverlapPredator format: dict with arrays
        # Sanity check - all arrays should have same length
        expected_keys = ['src', 'tgt', 'rot', 'trans', 'overlap']
        assert all(key in metadata_dict for key in expected_keys), f"Missing keys in metadata: {list(metadata_dict.keys())}"
        
        lengths = [len(metadata_dict[key]) for key in expected_keys]
        assert all(length == lengths[0] for length in lengths), f"Inconsistent lengths in metadata: {dict(zip(expected_keys, lengths))}"
        
        num_pairs = len(metadata_dict['src'])
        
        # Filter by overlap threshold and convert to annotations format
        self.annotations = []
        data_dir = os.path.join(self.data_root, 'data')
        for i in range(num_pairs):
            overlap = metadata_dict['overlap'][i]
            if self.overlap_threshold is None or overlap > self.overlap_threshold:
                # Extract scene names and ensure they match
                src_scene = metadata_dict['src'][i].split('/')[0]
                tgt_scene = metadata_dict['tgt'][i].split('/')[0]
                assert src_scene == tgt_scene, f"Scene names must match: src={src_scene}, tgt={tgt_scene}"
                
                annotation = {
                    'src_path': os.path.join(data_dir, metadata_dict['src'][i]),
                    'tgt_path': os.path.join(data_dir, metadata_dict['tgt'][i]),
                    'rotation': metadata_dict['rot'][i],  # (3, 3) numpy array
                    'translation': metadata_dict['trans'][i],  # (3,) numpy array
                    'overlap': overlap,
                    'scene_name': src_scene,  # Use verified scene name
                    'frag_id0': int(metadata_dict['src'][i].split('/')[-1].split('_')[-1].split('.')[0]),  # Extract fragment ID
                    'frag_id1': int(metadata_dict['tgt'][i].split('/')[-1].split('_')[-1].split('.')[0]),  # Extract fragment ID
                }
                self.annotations.append(annotation)
    
    def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        """Load a single datapoint from the dataset.
        
        Args:
            idx: Index of the datapoint to load
            
        Returns:
            Tuple of (inputs, labels, meta_info) dictionaries
        """
        # Get annotation
        annotation = self.annotations[idx]
        
        # Load point clouds  
        src_pc_tensor = load_point_cloud(annotation['src_path'])
        tgt_pc_tensor = load_point_cloud(annotation['tgt_path'])
        
        # Move to device
        if isinstance(src_pc_tensor, torch.Tensor):
            src_pc_tensor = src_pc_tensor.to(self.device)
        if isinstance(tgt_pc_tensor, torch.Tensor):
            tgt_pc_tensor = tgt_pc_tensor.to(self.device)
        
        # Create point cloud dictionaries
        src_pc = {
            'pos': src_pc_tensor,
            'feat': torch.ones((src_pc_tensor.shape[0], 1), dtype=torch.float32, device=self.device)
        }
        tgt_pc = {
            'pos': tgt_pc_tensor,
            'feat': torch.ones((tgt_pc_tensor.shape[0], 1), dtype=torch.float32, device=self.device)
        }
        
        # Create transformation matrix
        rotation = torch.tensor(annotation['rotation'], dtype=torch.float32, device=self.device)
        translation = torch.tensor(annotation['translation'], dtype=torch.float32, device=self.device)
        transform = torch.eye(4, dtype=torch.float32, device=self.device)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        
        # Get or compute correspondences with caching
        correspondences = self._get_cached_correspondences(annotation, src_pc['pos'], tgt_pc['pos'], transform)
        
        # Prepare inputs
        inputs = {
            'src_pc': src_pc,
            'tgt_pc': tgt_pc,
            'correspondences': correspondences,
        }
        
        # Prepare labels
        labels = {
            'transform': transform,
        }
        
        # Prepare meta_info (BaseDataset automatically adds 'idx')
        meta_info = {
            'src_path': annotation['src_path'],
            'tgt_path': annotation['tgt_path'],
            'scene_name': annotation.get('scene_name', 'unknown'),
            'overlap': annotation.get('overlap', 0.0),
            'src_frame': annotation.get('frag_id0', 0),
            'tgt_frame': annotation.get('frag_id1', 0),
        }
        
        return inputs, labels, meta_info
    
    def _get_cached_correspondences(
        self, 
        annotation: Dict[str, Any], 
        src_points: torch.Tensor, 
        tgt_points: torch.Tensor, 
        transform: torch.Tensor
    ) -> torch.Tensor:
        """Get correspondences with caching mechanism.
        
        Args:
            annotation: Annotation dictionary containing paths and metadata
            src_points: Source point cloud positions [M, 3]
            tgt_points: Target point cloud positions [N, 3]
            transform: Transformation matrix [4, 4]
            
        Returns:
            Correspondences tensor [K, 2]
        """
        # Create cache directory (sibling to data_root)
        cache_dir = os.path.join(os.path.dirname(self.data_root), f'{os.path.basename(self.data_root)}_correspondences_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create simple cache key from file basenames and radius
        src_name = os.path.basename(annotation['src_path']).split('.')[0]
        tgt_name = os.path.basename(annotation['tgt_path']).split('.')[0]
        cache_key = f"{src_name}_{tgt_name}_{self.matching_radius}"
        cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
        
        # Try to load from cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    correspondences = pickle.load(f)
                return torch.tensor(correspondences, dtype=torch.int64, device=self.device)
            except:
                # Cache file corrupted, recompute
                pass
        
        # Compute correspondences
        correspondences = get_correspondences(
            src_points=src_points,
            tgt_points=tgt_points,
            transform=transform,
            radius=self.matching_radius,
        )
        
        # Save to cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(correspondences.cpu().numpy(), f)
        except:
            # Cache write failed, but continue
            pass
        
        return correspondences