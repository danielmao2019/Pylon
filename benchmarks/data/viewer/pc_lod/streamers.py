"""Point cloud streamers for synthetic and real datasets."""

from typing import Dict, List, Any, Iterator
import sys
import os
import torch
import numpy as np
import random
from abc import ABC, abstractmethod

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, project_root)

from data.datasets.change_detection_datasets.bi_temporal.urb3dcd_dataset import Urb3DCDDataset
from data.datasets.change_detection_datasets.bi_temporal.slpccd_dataset import SLPCCDDataset  
from data.datasets.pcr_datasets.kitti_dataset import KITTIDataset

from .types import PointCloudSample


class PointCloudStreamer(ABC):
    """Abstract base class for streaming point clouds."""
    
    @abstractmethod
    def stream_point_clouds(self, num_samples: int) -> Iterator[PointCloudSample]:
        """Stream point cloud samples."""
        pass


class SyntheticPointCloudStreamer(PointCloudStreamer):
    """Streams synthetic point clouds with various configurations."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        torch.manual_seed(seed)
    
    def _generate_sphere(self, num_points: int, spatial_size: float, density_factor: float) -> torch.Tensor:
        """Generate points in a spherical distribution."""
        points = torch.randn(num_points, 3)
        points = points / torch.norm(points, dim=1, keepdim=True)
        radii = torch.pow(torch.rand(num_points), 1/3) * density_factor
        points = points * radii.unsqueeze(1)
        return points * spatial_size
    
    def _generate_cube(self, num_points: int, spatial_size: float, density_factor: float) -> torch.Tensor:
        """Generate points in a cubic distribution."""
        points = (torch.rand(num_points, 3) - 0.5) * 2 * spatial_size
        if density_factor != 1.0:
            center_weight = torch.exp(-torch.norm(points, dim=1) * (2 - density_factor))
            keep_mask = torch.rand(num_points) < center_weight
            points = points[keep_mask]
            if len(points) < num_points:
                extra = (torch.rand(num_points - len(points), 3) - 0.5) * 2 * spatial_size
                points = torch.cat([points, extra])
            else:
                points = points[:num_points]
        return points
    
    def _generate_gaussian(self, num_points: int, spatial_size: float, density_factor: float) -> torch.Tensor:
        """Generate points in a Gaussian distribution."""
        std = spatial_size * density_factor
        return torch.randn(num_points, 3) * std
    
    def _generate_plane(self, num_points: int, spatial_size: float, density_factor: float) -> torch.Tensor:
        """Generate points on a plane with some thickness."""
        points = torch.zeros(num_points, 3)
        points[:, :2] = (torch.rand(num_points, 2) - 0.5) * 2 * spatial_size
        points[:, 2] = torch.randn(num_points) * spatial_size * 0.1 * density_factor
        return points
    
    def create_test_configurations(self) -> List[Dict[str, Any]]:
        """Create various test configurations for synthetic point clouds."""
        configs = []
        
        # Point count variations
        for num_points in [10000, 25000, 50000, 100000, 200000]:
            configs.append({
                'num_points': num_points,
                'spatial_size': 10.0,
                'shape': 'sphere',
                'density_factor': 1.0,
                'name': f'{num_points//1000}K_points',
                'category': 'point_count'
            })
        
        # Spatial size variations
        for size in [1.0, 5.0, 10.0, 20.0, 50.0]:
            configs.append({
                'num_points': 50000,
                'spatial_size': size,
                'shape': 'cube',
                'density_factor': 1.0,
                'name': f'size_{size}',
                'category': 'spatial_size'
            })
        
        # Shape variations
        for shape in ['sphere', 'cube', 'gaussian', 'plane']:
            configs.append({
                'num_points': 50000,
                'spatial_size': 10.0,
                'shape': shape,
                'density_factor': 1.0,
                'name': shape,
                'category': 'shape'
            })
        
        # Density variations
        for density in [0.5, 1.0, 1.5, 2.0]:
            configs.append({
                'num_points': 50000,
                'spatial_size': 10.0,
                'shape': 'gaussian',
                'density_factor': density,
                'name': f'density_{density}',
                'category': 'density'
            })
        
        return configs
    
    def stream_point_clouds(self, num_samples: int = None) -> Iterator[PointCloudSample]:
        """Stream synthetic point cloud samples."""
        configs = self.create_test_configurations()
        
        if num_samples:
            configs = configs[:num_samples]
        
        generators = {
            'sphere': self._generate_sphere,
            'cube': self._generate_cube,
            'gaussian': self._generate_gaussian,
            'plane': self._generate_plane
        }
        
        for config in configs:
            points = generators[config['shape']](
                config['num_points'], 
                config['spatial_size'], 
                config['density_factor']
            )
            colors = torch.rand(config['num_points'], 3)
            
            yield PointCloudSample(
                name=config['name'],
                points=points,
                colors=colors,
                source='synthetic',
                metadata=config
            )


class RealDataPointCloudStreamer(PointCloudStreamer):
    """Streams point clouds from real datasets."""
    
    def __init__(self, data_root: str, seed: int = 42):
        self.data_root = data_root
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def _load_urb3dcd_samples(self, num_samples: int) -> List[PointCloudSample]:
        """Load samples from URB3DCD dataset."""
        dataset = Urb3DCDDataset(
            data_root=self.data_root,
            split='train',
            patched=False,
            radius=20  # Required when patched=False
        )
        
        total_samples = min(len(dataset), num_samples * 5)
        sample_indices = random.sample(range(total_samples), min(total_samples, num_samples))
        
        samples = []
        for idx in sample_indices:
            inputs, labels, meta_info = dataset[idx]
            
            pc_1 = inputs['pc_1']['pos']
            pc_2 = inputs['pc_2']['pos']
            
            colors_1 = torch.ones(pc_1.shape[0], 3, dtype=torch.float32)
            colors_2 = torch.ones(pc_2.shape[0], 3, dtype=torch.float32)
            
            samples.extend([
                PointCloudSample(f'urb3dcd_dp{idx}_pc1', pc_1, colors_1, 'urb3dcd', {'datapoint': idx, 'pc_type': 'pc1'}),
                PointCloudSample(f'urb3dcd_dp{idx}_pc2', pc_2, colors_2, 'urb3dcd', {'datapoint': idx, 'pc_type': 'pc2'})
            ])
            
            if len(samples) >= num_samples * 2:
                break
        
        return samples[:num_samples * 2]
    
    def _load_slpccd_samples(self, num_samples: int) -> List[PointCloudSample]:
        """Load samples from SLPCCD dataset."""
        dataset = SLPCCDDataset(
            data_root=self.data_root,
            split='train',
            use_hierarchy=False
        )
        
        total_samples = min(len(dataset), num_samples * 5)
        sample_indices = random.sample(range(total_samples), min(total_samples, num_samples))
        
        samples = []
        for idx in sample_indices:
            inputs, labels, meta_info = dataset[idx]
            
            if 'xyz' in inputs['pc_1']:
                pc_1 = inputs['pc_1']['xyz']
                pc_2 = inputs['pc_2']['xyz']
            else:
                pc_1 = inputs['pc_1']['pos'] if 'pos' in inputs['pc_1'] else inputs['pc_1']['xyz']
                pc_2 = inputs['pc_2']['pos'] if 'pos' in inputs['pc_2'] else inputs['pc_2']['xyz']
            
            if 'feat' in inputs['pc_1'] and inputs['pc_1']['feat'].shape[1] >= 3:
                colors_1 = inputs['pc_1']['feat'][:, :3]
                colors_2 = inputs['pc_2']['feat'][:, :3]
            else:
                colors_1 = torch.ones(pc_1.shape[0], 3, dtype=torch.float32)
                colors_2 = torch.ones(pc_2.shape[0], 3, dtype=torch.float32)
            
            samples.extend([
                PointCloudSample(f'slpccd_dp{idx}_pc1', pc_1, colors_1, 'slpccd', {'datapoint': idx, 'pc_type': 'pc1'}),
                PointCloudSample(f'slpccd_dp{idx}_pc2', pc_2, colors_2, 'slpccd', {'datapoint': idx, 'pc_type': 'pc2'})
            ])
            
            if len(samples) >= num_samples * 2:
                break
        
        return samples[:num_samples * 2]
    
    def _load_kitti_samples(self, num_samples: int) -> List[PointCloudSample]:
        """Load samples from KITTI dataset."""
        dataset = KITTIDataset(
            data_root=self.data_root,
            split='train'
        )
        
        total_samples = min(len(dataset), num_samples * 5)
        sample_indices = random.sample(range(total_samples), min(total_samples, num_samples))
        
        samples = []
        for idx in sample_indices:
            inputs, labels, meta_info = dataset[idx]
            
            src_pc = inputs['src_pc']['pos']
            tgt_pc = inputs['tgt_pc']['pos']
            
            if 'reflectance' in inputs['src_pc']:
                colors_src = inputs['src_pc']['reflectance'].repeat(1, 3)
                colors_tgt = inputs['tgt_pc']['reflectance'].repeat(1, 3)
            else:
                colors_src = torch.ones(src_pc.shape[0], 3, dtype=torch.float32)
                colors_tgt = torch.ones(tgt_pc.shape[0], 3, dtype=torch.float32)
            
            samples.extend([
                PointCloudSample(f'kitti_dp{idx}_src', src_pc, colors_src, 'kitti', {'datapoint': idx, 'pc_type': 'src'}),
                PointCloudSample(f'kitti_dp{idx}_tgt', tgt_pc, colors_tgt, 'kitti', {'datapoint': idx, 'pc_type': 'tgt'})
            ])
            
            if len(samples) >= num_samples * 2:
                break
        
        return samples[:num_samples * 2]
    
    def stream_point_clouds(self, num_samples: int) -> Iterator[PointCloudSample]:
        """Stream real dataset point cloud samples."""
        datasets = ['urb3dcd', 'slpccd', 'kitti']
        loaders = {
            'urb3dcd': self._load_urb3dcd_samples,
            'slpccd': self._load_slpccd_samples,
            'kitti': self._load_kitti_samples
        }
        
        for dataset_name in datasets:
            print(f"Loading samples from {dataset_name.upper()}...")
            samples = loaders[dataset_name](num_samples)
            
            if samples:
                print(f"  Loaded {len(samples)} point clouds from {dataset_name}")
                for sample in samples:
                    yield sample
            else:
                print(f"  No samples loaded from {dataset_name}")