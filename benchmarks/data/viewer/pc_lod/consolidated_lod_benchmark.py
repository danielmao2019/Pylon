#!/usr/bin/env python3
"""Consolidated comprehensive LOD benchmark with visual reports and multiple test scenarios."""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, project_root)

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from pathlib import Path
import json
import gc
import random
from dataclasses import dataclass, asdict
from data.viewer.utils.point_cloud import create_point_cloud_figure
from data.viewer.utils.camera_lod import get_lod_manager

# Import the dataset classes
try:
    from data.datasets.change_detection_datasets.bi_temporal.urb3dcd_dataset import Urb3DCDDataset
    from data.datasets.change_detection_datasets.bi_temporal.slpccd_dataset import SLPCCDDataset  
    from data.datasets.pcr_datasets.kitti_dataset import KITTIDataset
    REAL_DATA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import dataset classes: {e}")
    REAL_DATA_AVAILABLE = False


@dataclass
class PointCloudConfig:
    """Configuration for point cloud generation."""
    num_points: int
    spatial_size: float  # Scale factor for point cloud extent
    shape: str  # 'sphere', 'cube', 'gaussian', 'plane'
    density_factor: float  # Controls local density variation
    name: str  # Human readable name


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config: PointCloudConfig
    camera_distance: float
    
    # Timing breakdowns (averaged across runs)
    no_lod_init_time: float
    no_lod_render_time: float
    no_lod_total_time: float
    
    lod_init_time: float
    lod_render_time: float
    lod_total_time: float
    
    # LOD information
    lod_level: int
    original_points: int
    final_points: int
    point_reduction_pct: float
    
    # Performance metrics
    speedup_ratio: float
    render_speedup: float
    initialization_overhead: float


@dataclass  
class RealDataBenchmarkResult:
    """Results from real dataset benchmark runs."""
    dataset_name: str
    camera_distance_group: str  # 'close', 'medium', 'far'
    camera_distance_value: float
    
    # Timing breakdowns (averaged across datapoints and camera poses)
    no_lod_total_time: float
    lod_total_time: float
    
    # LOD information (averaged)
    avg_lod_level: float
    avg_original_points: float
    avg_final_points: float
    avg_point_reduction_pct: float
    
    # Performance metrics
    speedup_ratio: float
    
    # Statistical info
    num_datapoints: int
    num_camera_poses: int


class PointCloudGenerator:
    """Generate different types of point clouds for benchmarking."""
    
    @staticmethod
    def generate_sphere(num_points: int, spatial_size: float, density_factor: float) -> torch.Tensor:
        """Generate points in a spherical distribution."""
        # Generate on unit sphere first
        points = torch.randn(num_points, 3)
        points = points / torch.norm(points, dim=1, keepdim=True)
        
        # Add radial variation based on density factor
        radii = torch.pow(torch.rand(num_points), 1/3) * density_factor
        points = points * radii.unsqueeze(1)
        
        return points * spatial_size
    
    @staticmethod
    def generate_cube(num_points: int, spatial_size: float, density_factor: float) -> torch.Tensor:
        """Generate points in a cubic distribution."""
        # Uniform distribution in cube
        points = (torch.rand(num_points, 3) - 0.5) * 2 * spatial_size
        
        # Add density variation
        if density_factor != 1.0:
            # Create denser regions towards center
            center_weight = torch.exp(-torch.norm(points, dim=1) * (2 - density_factor))
            keep_mask = torch.rand(num_points) < center_weight
            points = points[keep_mask]
            
            # Pad or trim to exact count
            if len(points) < num_points:
                extra = (torch.rand(num_points - len(points), 3) - 0.5) * 2 * spatial_size
                points = torch.cat([points, extra])
            else:
                points = points[:num_points]
        
        return points
    
    @staticmethod
    def generate_gaussian(num_points: int, spatial_size: float, density_factor: float) -> torch.Tensor:
        """Generate points in a Gaussian distribution."""
        std = spatial_size * density_factor
        return torch.randn(num_points, 3) * std
    
    @staticmethod
    def generate_plane(num_points: int, spatial_size: float, density_factor: float) -> torch.Tensor:
        """Generate points on a plane with some thickness."""
        # Points on XY plane with small Z variation
        points = torch.zeros(num_points, 3)
        points[:, :2] = (torch.rand(num_points, 2) - 0.5) * 2 * spatial_size
        points[:, 2] = torch.randn(num_points) * spatial_size * 0.1 * density_factor
        return points
    
    @classmethod
    def generate(cls, config: PointCloudConfig) -> Dict[str, torch.Tensor]:
        """Generate point cloud based on configuration."""
        generators = {
            'sphere': cls.generate_sphere,
            'cube': cls.generate_cube,
            'gaussian': cls.generate_gaussian,
            'plane': cls.generate_plane
        }
        
        if config.shape not in generators:
            raise ValueError(f"Unknown shape: {config.shape}")
        
        points = generators[config.shape](config.num_points, config.spatial_size, config.density_factor)
        colors = torch.rand(config.num_points, 3)
        
        return {'pos': points, 'rgb': colors}


class RealDataLoader:
    """Loads point clouds from real datasets for benchmarking."""
    
    def __init__(self, data_root: str = None):
        self.data_root = data_root
        
    def load_dataset_samples(self, dataset_name: str, num_samples: int = 10, seed: int = 42) -> List[Dict[str, Any]]:
        """Load random samples from a real dataset.
        
        Args:
            dataset_name: Name of dataset ('urb3dcd', 'slpccd', 'kitti')
            num_samples: Number of datapoints to sample
            seed: Random seed for reproducible sampling
            
        Returns:
            List of dictionaries containing point cloud data
        """
        if not REAL_DATA_AVAILABLE:
            print(f"Warning: Real datasets not available, skipping {dataset_name}")
            return []
            
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        try:
            if dataset_name == 'urb3dcd':
                return self._load_urb3dcd_samples(num_samples)
            elif dataset_name == 'slpccd':
                return self._load_slpccd_samples(num_samples)
            elif dataset_name == 'kitti':
                return self._load_kitti_samples(num_samples)
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
        except Exception as e:
            print(f"Error loading {dataset_name} dataset: {e}")
            return []
    
    def _load_urb3dcd_samples(self, num_samples: int) -> List[Dict[str, Any]]:
        """Load samples from URB3DCD dataset."""
        if self.data_root is None:
            print("Warning: No data_root provided, cannot load URB3DCD dataset")
            return []
            
        try:
            dataset = Urb3DCDDataset(
                data_root=self.data_root,
                split='train',
                patched=False,  # Load full point clouds
                deterministic_seed=42
            )
            
            # Randomly sample datapoint indices
            total_samples = min(len(dataset), num_samples * 5)  # Sample more to account for loading failures
            sample_indices = random.sample(range(total_samples), min(total_samples, num_samples))
            
            samples = []
            for idx in sample_indices:
                try:
                    inputs, labels, meta_info = dataset[idx]
                    
                    # Extract both point clouds from each datapoint
                    pc_1 = inputs['pc_1']['pos']
                    pc_2 = inputs['pc_2']['pos']
                    
                    # Create color data (use ones if no color available)
                    colors_1 = torch.ones(pc_1.shape[0], 3, dtype=torch.float32)
                    colors_2 = torch.ones(pc_2.shape[0], 3, dtype=torch.float32)
                    
                    samples.append({
                        'name': f'urb3dcd_datapoint_{idx}_pc1',
                        'points': pc_1,
                        'colors': colors_1,
                        'source': 'urb3dcd'
                    })
                    
                    samples.append({
                        'name': f'urb3dcd_datapoint_{idx}_pc2', 
                        'points': pc_2,
                        'colors': colors_2,
                        'source': 'urb3dcd'
                    })
                    
                    if len(samples) >= num_samples * 2:  # Two point clouds per datapoint
                        break
                        
                except Exception as e:
                    print(f"Failed to load URB3DCD sample {idx}: {e}")
                    continue
            
            return samples[:num_samples * 2]  # Return up to num_samples * 2 point clouds
            
        except Exception as e:
            print(f"Failed to initialize URB3DCD dataset: {e}")
            return []
    
    def _load_slpccd_samples(self, num_samples: int) -> List[Dict[str, Any]]:
        """Load samples from SLPCCD dataset."""
        if self.data_root is None:
            print("Warning: No data_root provided, cannot load SLPCCD dataset")
            return []
            
        try:
            dataset = SLPCCDDataset(
                data_root=self.data_root,
                split='train',
                use_hierarchy=False,  # Disable hierarchy for simple point cloud access
                deterministic_seed=42
            )
            
            # Randomly sample datapoint indices
            total_samples = min(len(dataset), num_samples * 5)
            sample_indices = random.sample(range(total_samples), min(total_samples, num_samples))
            
            samples = []
            for idx in sample_indices:
                try:
                    inputs, labels, meta_info = dataset[idx]
                    
                    # Extract both point clouds
                    if 'xyz' in inputs['pc_1']:
                        pc_1 = inputs['pc_1']['xyz']
                        pc_2 = inputs['pc_2']['xyz']
                    else:
                        pc_1 = inputs['pc_1']['pos'] if 'pos' in inputs['pc_1'] else inputs['pc_1']['xyz']
                        pc_2 = inputs['pc_2']['pos'] if 'pos' in inputs['pc_2'] else inputs['pc_2']['xyz']
                    
                    # Extract features as colors if available
                    if 'feat' in inputs['pc_1'] and inputs['pc_1']['feat'].shape[1] >= 3:
                        colors_1 = inputs['pc_1']['feat'][:, :3]
                        colors_2 = inputs['pc_2']['feat'][:, :3]
                    else:
                        colors_1 = torch.ones(pc_1.shape[0], 3, dtype=torch.float32)
                        colors_2 = torch.ones(pc_2.shape[0], 3, dtype=torch.float32)
                    
                    samples.append({
                        'name': f'slpccd_datapoint_{idx}_pc1',
                        'points': pc_1,
                        'colors': colors_1,
                        'source': 'slpccd'
                    })
                    
                    samples.append({
                        'name': f'slpccd_datapoint_{idx}_pc2',
                        'points': pc_2, 
                        'colors': colors_2,
                        'source': 'slpccd'
                    })
                    
                    if len(samples) >= num_samples * 2:
                        break
                        
                except Exception as e:
                    print(f"Failed to load SLPCCD sample {idx}: {e}")
                    continue
            
            return samples[:num_samples * 2]
            
        except Exception as e:
            print(f"Failed to initialize SLPCCD dataset: {e}")
            return []
    
    def _load_kitti_samples(self, num_samples: int) -> List[Dict[str, Any]]:
        """Load samples from KITTI dataset."""
        if self.data_root is None:
            print("Warning: No data_root provided, cannot load KITTI dataset")
            return []
            
        try:
            dataset = KITTIDataset(
                data_root=self.data_root,
                split='train',
                deterministic_seed=42
            )
            
            # Randomly sample datapoint indices
            total_samples = min(len(dataset), num_samples * 5)
            sample_indices = random.sample(range(total_samples), min(total_samples, num_samples))
            
            samples = []
            for idx in sample_indices:
                try:
                    inputs, labels, meta_info = dataset[idx]
                    
                    # Extract source and target point clouds
                    src_pc = inputs['src_pc']['pos']
                    tgt_pc = inputs['tgt_pc']['pos']
                    
                    # Use reflectance as color if available, otherwise use ones
                    if 'reflectance' in inputs['src_pc']:
                        src_refl = inputs['src_pc']['reflectance']
                        tgt_refl = inputs['tgt_pc']['reflectance']
                        # Convert single-channel reflectance to 3-channel color
                        colors_src = src_refl.repeat(1, 3)
                        colors_tgt = tgt_refl.repeat(1, 3)
                    else:
                        colors_src = torch.ones(src_pc.shape[0], 3, dtype=torch.float32)
                        colors_tgt = torch.ones(tgt_pc.shape[0], 3, dtype=torch.float32)
                    
                    samples.append({
                        'name': f'kitti_datapoint_{idx}_src',
                        'points': src_pc,
                        'colors': colors_src,
                        'source': 'kitti'
                    })
                    
                    samples.append({
                        'name': f'kitti_datapoint_{idx}_tgt',
                        'points': tgt_pc,
                        'colors': colors_tgt,
                        'source': 'kitti'
                    })
                    
                    if len(samples) >= num_samples * 2:
                        break
                        
                except Exception as e:
                    print(f"Failed to load KITTI sample {idx}: {e}")
                    continue
            
            return samples[:num_samples * 2]
            
        except Exception as e:
            print(f"Failed to initialize KITTI dataset: {e}")
            return []


class CameraPoseGenerator:
    """Generates camera poses for benchmarking point cloud rendering."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        
    def generate_camera_poses(self, point_cloud: torch.Tensor, num_poses_per_distance: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Generate camera poses at different distances from point cloud.
        
        Args:
            point_cloud: Point cloud tensor of shape (N, 3)
            num_poses_per_distance: Number of camera poses to generate per distance group
            
        Returns:
            Dictionary with distance groups as keys and lists of camera poses as values
        """
        # Calculate point cloud center and bounds
        pc_center = point_cloud.mean(dim=0).numpy()
        pc_bounds = (point_cloud.min().item(), point_cloud.max().item())
        pc_size = pc_bounds[1] - pc_bounds[0]
        
        # Define distance groups based on point cloud size
        distance_groups = {
            'close': pc_size * 0.5,      # Close viewing
            'medium': pc_size * 2.0,     # Medium distance  
            'far': pc_size * 5.0         # Far viewing (should trigger LOD)
        }
        
        camera_poses = {}
        
        for group_name, base_distance in distance_groups.items():
            poses = []
            
            for i in range(num_poses_per_distance):
                # Generate camera position facing toward the point cloud
                # Start with a view direction toward the center
                theta = self.rng.uniform(0, 2 * np.pi)  # Azimuth angle
                phi = self.rng.uniform(np.pi/6, np.pi/3)  # Elevation angle (avoid top-down)
                
                # Convert spherical to cartesian coordinates
                x = base_distance * np.sin(phi) * np.cos(theta)
                y = base_distance * np.sin(phi) * np.sin(theta) 
                z = base_distance * np.cos(phi)
                
                # Add some translation to the side (perpendicular to view direction)
                # This simulates the case where point cloud is not exactly centered
                side_offset = self.rng.uniform(-pc_size * 0.3, pc_size * 0.3, 3)
                side_offset[2] *= 0.2  # Less vertical offset
                
                camera_pos = pc_center + np.array([x, y, z]) + side_offset
                
                # Camera always looks toward the point cloud center
                camera_state = {
                    'eye': {
                        'x': float(camera_pos[0]),
                        'y': float(camera_pos[1]),
                        'z': float(camera_pos[2])
                    },
                    'center': {
                        'x': float(pc_center[0]), 
                        'y': float(pc_center[1]),
                        'z': float(pc_center[2])
                    },
                    'up': {'x': 0, 'y': 0, 'z': 1}  # Z-up convention
                }
                
                poses.append({
                    'camera_state': camera_state,
                    'distance': base_distance,
                    'pose_id': i
                })
            
            camera_poses[group_name] = poses
        
        return camera_poses


class ConsolidatedLODBenchmark:
    """Comprehensive LOD performance benchmark with multiple test modes."""
    
    def __init__(self, output_dir: str = "benchmarks/data/viewer/pc_lod/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear LOD cache before benchmarking
        lod_manager = get_lod_manager()
        lod_manager.clear_cache()
        
        # Benchmark configuration
        self.num_runs = 3
        self.camera_distances = [0.1, 2.0, 10.0]  # Close, medium, far
        
        # Real data configuration
        self.real_data_loader = RealDataLoader()
        self.camera_pose_generator = CameraPoseGenerator(seed=42)
        
    def create_point_cloud_configs(self) -> Dict[str, List[PointCloudConfig]]:
        """Create different point cloud configurations for testing."""
        configs = {
            # Vary number of points (fixed shape, size, density)
            'num_points': [
                PointCloudConfig(1000, 10.0, 'sphere', 1.0, '1K Points'),
                PointCloudConfig(5000, 10.0, 'sphere', 1.0, '5K Points'),
                PointCloudConfig(10000, 10.0, 'sphere', 1.0, '10K Points'),
                PointCloudConfig(25000, 10.0, 'sphere', 1.0, '25K Points'),
                PointCloudConfig(50000, 10.0, 'sphere', 1.0, '50K Points'),
                PointCloudConfig(100000, 10.0, 'sphere', 1.0, '100K Points'),
                PointCloudConfig(200000, 10.0, 'sphere', 1.0, '200K Points'),
            ],
            
            # Vary spatial size (fixed points, shape, density)  
            'spatial_size': [
                PointCloudConfig(50000, 1.0, 'cube', 1.0, 'Size 1.0'),
                PointCloudConfig(50000, 5.0, 'cube', 1.0, 'Size 5.0'),
                PointCloudConfig(50000, 10.0, 'cube', 1.0, 'Size 10.0'),
                PointCloudConfig(50000, 20.0, 'cube', 1.0, 'Size 20.0'),
                PointCloudConfig(50000, 50.0, 'cube', 1.0, 'Size 50.0'),
            ],
            
            # Vary density (fixed points, shape, size)
            'density': [
                PointCloudConfig(50000, 10.0, 'gaussian', 0.5, 'Low Density'),
                PointCloudConfig(50000, 10.0, 'gaussian', 1.0, 'Medium Density'),
                PointCloudConfig(50000, 10.0, 'gaussian', 1.5, 'High Density'),
                PointCloudConfig(50000, 10.0, 'gaussian', 2.0, 'Very High Density'),
            ],
            
            # Vary shape (fixed points, size, density)
            'shape': [
                PointCloudConfig(50000, 10.0, 'sphere', 1.0, 'Sphere'),
                PointCloudConfig(50000, 10.0, 'cube', 1.0, 'Cube'),
                PointCloudConfig(50000, 10.0, 'gaussian', 1.0, 'Gaussian'),
                PointCloudConfig(50000, 10.0, 'plane', 1.0, 'Plane'),
            ]
        }
        
        return configs
    
    def benchmark_single_config(self, config: PointCloudConfig, camera_distance: float) -> BenchmarkResult:
        """Benchmark a single point cloud configuration."""
        print(f"  📊 Testing {config.name} at distance {camera_distance:.1f}")
        
        # Generate point cloud
        pc_data = PointCloudGenerator.generate(config)
        points = pc_data['pos']
        colors = pc_data['rgb']
        
        # Create camera state
        pc_center = points.mean(dim=0).numpy()
        pc_bounds = (points.min().item(), points.max().item())
        pc_size = pc_bounds[1] - pc_bounds[0]
        actual_distance = camera_distance * pc_size
        
        camera_state = {
            'eye': {
                'x': pc_center[0] + actual_distance,
                'y': pc_center[1] + actual_distance,
                'z': pc_center[2] + actual_distance
            },
            'center': {'x': pc_center[0], 'y': pc_center[1], 'z': pc_center[2]},
            'up': {'x': 0, 'y': 0, 'z': 1}
        }
        
        # Benchmark WITHOUT LOD
        no_lod_times = []
        for run in range(self.num_runs):
            gc.collect()  # Clean memory before each run
            
            init_start = time.perf_counter()
            render_start = time.perf_counter()
            fig = create_point_cloud_figure(
                points=points,
                colors=colors,
                title=f"No LOD {run}",
                camera_state=camera_state,
                lod_enabled=False
            )
            render_end = time.perf_counter()
            
            no_lod_times.append({
                'init_time': render_start - init_start,
                'render_time': render_end - render_start,
                'total_time': render_end - init_start
            })
        
        # Average no-LOD times
        avg_no_lod = {
            'init_time': np.mean([t['init_time'] for t in no_lod_times]),
            'render_time': np.mean([t['render_time'] for t in no_lod_times]),
            'total_time': np.mean([t['total_time'] for t in no_lod_times])
        }
        
        # Benchmark WITH LOD
        lod_times = []
        lod_info = {'level': 0, 'final_points': config.num_points}
        
        for run in range(self.num_runs):
            gc.collect()  # Clean memory before each run
            
            init_start = time.perf_counter()
            render_start = time.perf_counter()
            fig = create_point_cloud_figure(
                points=points,
                colors=colors,
                title=f"LOD {run}",
                camera_state=camera_state,
                lod_enabled=True,
                point_cloud_id=f"benchmark_{config.name}_{camera_distance}"
            )
            render_end = time.perf_counter()
            
            # Extract LOD info from first run
            if run == 0:
                title = fig.layout.title.text
                if "LOD" in title:
                    try:
                        lod_part = title.split("(LOD ")[1].split(":")[0].strip()
                        lod_info['level'] = int(lod_part)
                        points_part = title.split(": ")[1].split("/")[0]
                        lod_info['final_points'] = int(points_part.replace(",", ""))
                    except:
                        pass
            
            lod_times.append({
                'init_time': render_start - init_start,
                'render_time': render_end - render_start,
                'total_time': render_end - init_start
            })
        
        # Average LOD times
        avg_lod = {
            'init_time': np.mean([t['init_time'] for t in lod_times]),
            'render_time': np.mean([t['render_time'] for t in lod_times]),
            'total_time': np.mean([t['total_time'] for t in lod_times])
        }
        
        # Calculate metrics
        point_reduction_pct = (config.num_points - lod_info['final_points']) / config.num_points * 100
        speedup_ratio = avg_no_lod['total_time'] / avg_lod['total_time'] if avg_lod['total_time'] > 0 else 1.0
        render_speedup = avg_no_lod['render_time'] / avg_lod['render_time'] if avg_lod['render_time'] > 0 else 1.0
        
        # Calculate initialization overhead (first run vs subsequent runs)
        if len(lod_times) > 1:
            init_overhead = lod_times[0]['total_time'] - np.mean([t['total_time'] for t in lod_times[1:]])
        else:
            init_overhead = 0.0
        
        return BenchmarkResult(
            config=config,
            camera_distance=camera_distance,
            no_lod_init_time=avg_no_lod['init_time'],
            no_lod_render_time=avg_no_lod['render_time'],
            no_lod_total_time=avg_no_lod['total_time'],
            lod_init_time=avg_lod['init_time'],
            lod_render_time=avg_lod['render_time'],
            lod_total_time=avg_lod['total_time'],
            lod_level=lod_info['level'],
            original_points=config.num_points,
            final_points=lod_info['final_points'],
            point_reduction_pct=point_reduction_pct,
            speedup_ratio=speedup_ratio,
            render_speedup=render_speedup,
            initialization_overhead=max(0, init_overhead)
        )
    
    def run_quick_benchmark(self) -> Dict[str, List[float]]:
        """Run a quick benchmark for immediate feedback."""
        print("🚀 Quick LOD Benchmark")
        
        # Test configurations
        point_counts = [10000, 50000, 100000]
        
        results = {
            'point_counts': [],
            'no_lod_times': [],
            'lod_times': [],
            'speedups': [],
            'point_reductions': []
        }
        
        # Far camera to trigger LOD
        far_camera = {'eye': {'x': 50, 'y': 50, 'z': 50}}
        
        for num_points in point_counts:
            print(f"  📊 Testing {num_points:,} points")
            
            # Generate test data
            points = torch.randn(num_points, 3) * 10
            colors = torch.rand(num_points, 3)
            
            # Test without LOD
            start = time.perf_counter()
            fig1 = create_point_cloud_figure(
                points=points,
                colors=colors,
                title="No LOD",
                camera_state=far_camera,
                lod_enabled=False
            )
            no_lod_time = time.perf_counter() - start
            
            # Test with LOD
            start = time.perf_counter()
            fig2 = create_point_cloud_figure(
                points=points,
                colors=colors,
                title="LOD Test",
                camera_state=far_camera,
                lod_enabled=True,
                point_cloud_id=f"test_{num_points}"
            )
            lod_time = time.perf_counter() - start
            
            # Extract results
            lod_title = fig2.layout.title.text
            final_points = num_points
            
            if "LOD" in lod_title:
                try:
                    points_part = lod_title.split(": ")[1].split("/")[0]
                    final_points = int(points_part.replace(",", ""))
                except:
                    pass
            
            speedup = no_lod_time / lod_time if lod_time > 0 else 1.0
            reduction = (num_points - final_points) / num_points * 100
            
            print(f"    No LOD: {no_lod_time:.3f}s")
            print(f"    LOD: {lod_time:.3f}s ({final_points:,} points)")
            print(f"    Speedup: {speedup:.2f}x, Reduction: {reduction:.1f}%")
            
            # Store results
            results['point_counts'].append(num_points)
            results['no_lod_times'].append(no_lod_time)
            results['lod_times'].append(lod_time)
            results['speedups'].append(speedup)
            results['point_reductions'].append(reduction)
        
        return results
    
    def run_comprehensive_benchmark_suite(self) -> Dict[str, List[BenchmarkResult]]:
        """Run complete comprehensive benchmark suite."""
        print("🚀 Starting comprehensive LOD benchmark suite...")
        
        configs = self.create_point_cloud_configs()
        all_results = {}
        
        for config_type, config_list in configs.items():
            print(f"\n📋 Testing {config_type} variations...")
            results = []
            
            for config in config_list:
                # Test at far distance where LOD should have most impact
                far_distance = 10.0
                result = self.benchmark_single_config(config, far_distance)
                results.append(result)
            
            all_results[config_type] = results
        
        return all_results
    
    def run_detailed_distance_benchmark(self) -> Dict[str, List[float]]:
        """Run detailed benchmark across multiple camera distances and point counts."""
        print("🚀 Running detailed distance-based benchmark...")
        
        point_counts = [1000, 10000, 50000, 100000, 200000]
        camera_distances = [0.1, 1.0, 3.0, 10.0]  # Close to far (normalized)
        
        results = {
            'point_counts': [],
            'camera_distances': [],
            'lod_disabled_times': [],
            'lod_enabled_times': [],
            'lod_levels': [],
            'final_point_counts': [],
            'speedup_ratios': [],
            'initialization_overhead': []
        }
        
        for point_count in point_counts:
            print(f"\n🔍 Testing with {point_count:,} points...")
            
            # Create test data
            points = torch.randn(point_count, 3, dtype=torch.float32) * 10
            colors = torch.rand(point_count, 3, dtype=torch.float32)
            
            for camera_distance in camera_distances:
                print(f"  📍 Camera distance: {camera_distance}")
                
                # Create camera state based on distance
                pc_center = points.mean(dim=0).numpy()
                pc_bounds = (points.min().item(), points.max().item())
                pc_size = pc_bounds[1] - pc_bounds[0]
                actual_distance = camera_distance * pc_size
                
                camera_state = {
                    'eye': {
                        'x': pc_center[0] + actual_distance,
                        'y': pc_center[1] + actual_distance, 
                        'z': pc_center[2] + actual_distance
                    },
                    'center': {'x': pc_center[0], 'y': pc_center[1], 'z': pc_center[2]},
                    'up': {'x': 0, 'y': 0, 'z': 1}
                }
                
                # Benchmark WITHOUT LOD
                lod_disabled_times = []
                for run in range(self.num_runs):
                    gc.collect()
                    
                    start_time = time.perf_counter()
                    fig_no_lod = create_point_cloud_figure(
                        points=points,
                        colors=colors,
                        title=f"No LOD Test {run}",
                        camera_state=camera_state,
                        lod_enabled=False,
                        point_cloud_id=None
                    )
                    end_time = time.perf_counter()
                    
                    runtime = end_time - start_time
                    lod_disabled_times.append(runtime)
                
                avg_no_lod_time = np.mean(lod_disabled_times)
                
                # Benchmark WITH LOD  
                lod_enabled_times = []
                lod_level = None
                final_point_count = point_count
                
                for run in range(self.num_runs):
                    gc.collect()
                    
                    start_time = time.perf_counter()
                    fig_with_lod = create_point_cloud_figure(
                        points=points,
                        colors=colors,
                        title=f"LOD Test {run}",
                        camera_state=camera_state,
                        lod_enabled=True,
                        point_cloud_id=f"benchmark_{point_count}_{camera_distance}"
                    )
                    end_time = time.perf_counter()
                    
                    runtime = end_time - start_time
                    lod_enabled_times.append(runtime)
                    
                    # Extract LOD info from title on first run
                    if run == 0:
                        title = fig_with_lod.layout.title.text
                        if "LOD" in title:
                            try:
                                lod_part = title.split("(LOD ")[1].split(":")[0]
                                lod_level = int(lod_part.split()[0])
                                points_part = title.split(": ")[1].split("/")[0]
                                final_point_count = int(points_part.replace(",", ""))
                            except:
                                lod_level = 0
                                final_point_count = point_count
                        else:
                            lod_level = 0
                            final_point_count = point_count
                
                avg_lod_time = np.mean(lod_enabled_times)
                
                # Calculate metrics
                speedup_ratio = avg_no_lod_time / avg_lod_time if avg_lod_time > 0 else 1.0
                
                # Estimate initialization overhead (first run vs subsequent runs)
                if len(lod_enabled_times) > 1:
                    init_overhead = lod_enabled_times[0] - np.mean(lod_enabled_times[1:])
                else:
                    init_overhead = 0.0
                    
                # Store results
                results['point_counts'].append(point_count)
                results['camera_distances'].append(camera_distance)
                results['lod_disabled_times'].append(avg_no_lod_time)
                results['lod_enabled_times'].append(avg_lod_time)
                results['lod_levels'].append(lod_level or 0)
                results['final_point_counts'].append(final_point_count)
                results['speedup_ratios'].append(speedup_ratio)
                results['initialization_overhead'].append(max(0, init_overhead))
                
                print(f"    📊 Results:")
                print(f"      No LOD: {avg_no_lod_time:.4f}s")
                print(f"      With LOD: {avg_lod_time:.4f}s (LOD {lod_level}, {final_point_count:,} points)")
                print(f"      Speedup: {speedup_ratio:.2f}x")
        
        return results
    
    def run_real_data_benchmark(self, data_root: str = None, num_samples: int = 10, 
                               num_poses_per_distance: int = 3) -> Dict[str, List[RealDataBenchmarkResult]]:
        """Run LOD benchmark on real datasets.
        
        Args:
            data_root: Path to dataset root directory
            num_samples: Number of datapoints to sample from each dataset
            num_poses_per_distance: Number of camera poses per distance group
            
        Returns:
            Dictionary with dataset names as keys and benchmark results as values
        """
        if not REAL_DATA_AVAILABLE:
            print("⚠️ Real datasets not available, skipping real data benchmark")
            return {}
            
        print("🏗️ Running real data benchmark...")
        
        # Set data root for loader
        if data_root:
            self.real_data_loader.data_root = data_root
        
        datasets = ['urb3dcd', 'slpccd', 'kitti']
        all_results = {}
        
        for dataset_name in datasets:
            print(f"\n📊 Benchmarking {dataset_name.upper()} dataset...")
            
            # Load samples from dataset
            samples = self.real_data_loader.load_dataset_samples(dataset_name, num_samples)
            
            if not samples:
                print(f"  ⚠️ No samples loaded from {dataset_name}, skipping...")
                continue
            
            print(f"  📦 Loaded {len(samples)} point clouds from {dataset_name}")
            
            # Run benchmark for this dataset
            dataset_results = self._benchmark_real_dataset(dataset_name, samples, num_poses_per_distance)
            all_results[dataset_name] = dataset_results
        
        return all_results
    
    def _benchmark_real_dataset(self, dataset_name: str, samples: List[Dict[str, Any]], 
                               num_poses_per_distance: int) -> List[RealDataBenchmarkResult]:
        """Benchmark a single real dataset.
        
        Args:
            dataset_name: Name of the dataset
            samples: List of point cloud samples
            num_poses_per_distance: Number of camera poses per distance group
            
        Returns:
            List of benchmark results grouped by camera distance
        """
        # Group results by distance
        distance_groups = {'close': [], 'medium': [], 'far': []}
        
        for sample_idx, sample in enumerate(samples):
            print(f"    🔍 Processing sample {sample_idx + 1}/{len(samples)}: {sample['name']}")
            
            points = sample['points']
            colors = sample['colors']
            
            # Generate camera poses for this point cloud
            camera_poses = self.camera_pose_generator.generate_camera_poses(
                points, num_poses_per_distance
            )
            
            # Benchmark each distance group
            for distance_group, poses in camera_poses.items():
                group_results = []
                
                for pose in poses:
                    # Benchmark this specific pose
                    no_lod_times = []
                    lod_times = []
                    lod_info = {'level': 0, 'final_points': len(points)}
                    
                    for run in range(self.num_runs):
                        gc.collect()  # Clean memory
                        
                        # Test without LOD
                        start_time = time.perf_counter()
                        fig_no_lod = create_point_cloud_figure(
                            points=points,
                            colors=colors,
                            title=f"No LOD {run}",
                            camera_state=pose['camera_state'],
                            lod_enabled=False
                        )
                        no_lod_time = time.perf_counter() - start_time
                        no_lod_times.append(no_lod_time)
                        
                        # Test with LOD
                        start_time = time.perf_counter()
                        fig_lod = create_point_cloud_figure(
                            points=points,
                            colors=colors,
                            title=f"LOD {run}",
                            camera_state=pose['camera_state'],
                            lod_enabled=True,
                            point_cloud_id=f"{dataset_name}_{sample_idx}_{distance_group}_{pose['pose_id']}"
                        )
                        lod_time = time.perf_counter() - start_time
                        lod_times.append(lod_time)
                        
                        # Extract LOD info on first run
                        if run == 0:
                            title = fig_lod.layout.title.text
                            if "LOD" in title:
                                try:
                                    lod_part = title.split("(LOD ")[1].split(":")[0].strip()
                                    lod_info['level'] = int(lod_part)
                                    points_part = title.split(": ")[1].split("/")[0]
                                    lod_info['final_points'] = int(points_part.replace(",", ""))
                                except:
                                    pass
                    
                    # Calculate averages for this pose
                    avg_no_lod_time = np.mean(no_lod_times)
                    avg_lod_time = np.mean(lod_times)
                    
                    group_results.append({
                        'no_lod_time': avg_no_lod_time,
                        'lod_time': avg_lod_time,
                        'lod_level': lod_info['level'],
                        'original_points': len(points),
                        'final_points': lod_info['final_points'],
                        'distance': pose['distance']
                    })
                
                # Store results for this distance group
                distance_groups[distance_group].extend(group_results)
        
        # Create aggregated results for each distance group
        aggregated_results = []
        
        for distance_group, group_data in distance_groups.items():
            if not group_data:
                continue
                
            # Calculate averages across all samples and poses in this distance group
            avg_no_lod_time = np.mean([r['no_lod_time'] for r in group_data])
            avg_lod_time = np.mean([r['lod_time'] for r in group_data])
            avg_lod_level = np.mean([r['lod_level'] for r in group_data])
            avg_original_points = np.mean([r['original_points'] for r in group_data])
            avg_final_points = np.mean([r['final_points'] for r in group_data])
            avg_distance = np.mean([r['distance'] for r in group_data])
            
            avg_point_reduction = (avg_original_points - avg_final_points) / avg_original_points * 100
            speedup_ratio = avg_no_lod_time / avg_lod_time if avg_lod_time > 0 else 1.0
            
            result = RealDataBenchmarkResult(
                dataset_name=dataset_name,
                camera_distance_group=distance_group,
                camera_distance_value=avg_distance,
                no_lod_total_time=avg_no_lod_time,
                lod_total_time=avg_lod_time,
                avg_lod_level=avg_lod_level,
                avg_original_points=avg_original_points,
                avg_final_points=avg_final_points,
                avg_point_reduction_pct=avg_point_reduction,
                speedup_ratio=speedup_ratio,
                num_datapoints=len(samples),
                num_camera_poses=len(group_data)
            )
            
            aggregated_results.append(result)
            
            print(f"      📈 {distance_group.capitalize()}: {speedup_ratio:.2f}x speedup, "
                  f"{avg_point_reduction:.1f}% reduction")
        
        return aggregated_results
    
    def create_real_data_plots(self, real_data_results: Dict[str, List[RealDataBenchmarkResult]]):
        """Create plots for real dataset benchmark results."""
        if not real_data_results:
            return
            
        print("📊 Creating real data performance plots...")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        colors = {'lod': '#2E86C1', 'no_lod': '#E74C3C'}
        
        # Create a figure for each dataset
        for dataset_name, results in real_data_results.items():
            if not results:
                continue
                
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.suptitle(f'LOD Performance on {dataset_name.upper()} Dataset', fontsize=14, fontweight='bold')
            
            # Extract data for plotting
            distance_groups = [r.camera_distance_group for r in results]
            no_lod_times = [r.no_lod_total_time for r in results]
            lod_times = [r.lod_total_time for r in results]
            speedups = [r.speedup_ratio for r in results]
            
            # Create paired bar chart
            x = np.arange(len(distance_groups))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, no_lod_times, width, label='No LOD', 
                          color=colors['no_lod'], alpha=0.8)
            bars2 = ax.bar(x + width/2, lod_times, width, label='With LOD', 
                          color=colors['lod'], alpha=0.8)
            
            # Add speedup annotations
            for i, (bar1, bar2, speedup) in enumerate(zip(bars1, bars2, speedups)):
                height = max(bar1.get_height(), bar2.get_height())
                ax.annotate(f'{speedup:.1f}x', 
                           xy=(i, height + height*0.05),
                           ha='center', va='bottom', fontweight='bold',
                           color='green' if speedup > 1.1 else 'gray')
            
            ax.set_xlabel('Camera Distance Group')
            ax.set_ylabel('Rendering Time (seconds)')
            ax.set_title(f'Average Performance Across {results[0].num_datapoints} Datapoints')
            ax.set_xticks(x)
            ax.set_xticklabels([g.capitalize() for g in distance_groups])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            avg_speedup = np.mean(speedups)
            avg_reduction = np.mean([r.avg_point_reduction_pct for r in results])
            
            stats_text = f"Average Speedup: {avg_speedup:.2f}x\nAverage Point Reduction: {avg_reduction:.1f}%"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.output_dir / f"real_data_{dataset_name}_performance.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  📈 Saved {dataset_name} plot: {plot_file}")
    
    def create_comprehensive_plots(self, results: Dict[str, List[BenchmarkResult]], 
                                  quick_results: Dict[str, List[float]] = None,
                                  distance_results: Dict[str, List[float]] = None,
                                  real_data_results: Dict[str, List[RealDataBenchmarkResult]] = None):
        """Create comprehensive performance visualization plots."""
        print("📊 Creating comprehensive performance visualization plots...")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        colors = {
            'lod': '#2E86C1',      # Blue for LOD
            'no_lod': '#E74C3C'    # Red for no LOD
        }
        
        # Create main comprehensive plot
        if results:
            self._create_main_performance_plots(results, colors)
        
        # Create quick benchmark plot
        if quick_results:
            self._create_quick_benchmark_plot(quick_results, colors)
        
        # Create distance analysis plot
        if distance_results:
            self._create_distance_analysis_plot(distance_results, colors)
        
        # Create real data plots
        if real_data_results:
            self.create_real_data_plots(real_data_results)
    
    def _create_main_performance_plots(self, results: Dict[str, List[BenchmarkResult]], colors: dict):
        """Create main comprehensive performance plots."""
        for config_type, result_list in results.items():
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'LOD Performance Analysis: {config_type.replace("_", " ").title()}', 
                        fontsize=16, fontweight='bold')
            
            # Extract data
            x_values = []
            x_labels = []
            no_lod_total = []
            lod_total = []
            speedups = []
            point_reductions = []
            final_points = []
            
            for i, result in enumerate(result_list):
                x_values.append(i)
                x_labels.append(result.config.name)
                no_lod_total.append(result.no_lod_total_time)
                lod_total.append(result.lod_total_time)
                speedups.append(result.speedup_ratio)
                point_reductions.append(result.point_reduction_pct)
                final_points.append(result.final_points)
            
            # Plot 1: Total time comparison
            ax1 = axes[0, 0]
            width = 0.35
            x_pos = np.arange(len(x_values))
            
            bars1 = ax1.bar(x_pos - width/2, no_lod_total, width, label='No LOD Total', 
                           color=colors['no_lod'], alpha=0.8)
            bars2 = ax1.bar(x_pos + width/2, lod_total, width, label='LOD Total', 
                           color=colors['lod'], alpha=0.8)
            
            ax1.set_xlabel('Configuration')
            ax1.set_ylabel('Total Time (seconds)')
            ax1.set_title('Total Rendering Time Comparison')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(x_labels, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Add speedup annotations
            for i, (bar1, bar2, speedup) in enumerate(zip(bars1, bars2, speedups)):
                height = max(bar1.get_height(), bar2.get_height())
                ax1.annotate(f'{speedup:.1f}x', 
                           xy=(i, height + height*0.05),
                           ha='center', va='bottom', fontweight='bold',
                           color='green' if speedup > 1.1 else 'gray')
            
            # Plot 2: Speedup ratios
            ax2 = axes[0, 1]
            bars = ax2.bar(x_values, speedups, color=[colors['lod'] if s > 1.1 else 'gray' for s in speedups], alpha=0.8)
            ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No Improvement')
            ax2.set_xlabel('Configuration')
            ax2.set_ylabel('Speedup Ratio')
            ax2.set_title('Performance Speedup (Higher is Better)')
            ax2.set_xticks(x_values)
            ax2.set_xticklabels(x_labels, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, speedup in zip(bars, speedups):
                height = bar.get_height()
                ax2.annotate(f'{speedup:.2f}x',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
            
            # Plot 3: Point reduction efficiency
            ax3 = axes[1, 0]
            scatter = ax3.scatter(point_reductions, speedups, c=final_points, cmap='viridis', 
                                s=100, alpha=0.7, edgecolors='black', linewidth=1)
            
            ax3.set_xlabel('Point Reduction (%)')
            ax3.set_ylabel('Speedup Ratio')
            ax3.set_title('LOD Efficiency: Point Reduction vs Speedup')
            ax3.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax3)
            cbar.set_label('Final Point Count')
            
            # Plot 4: Timing breakdown
            ax4 = axes[1, 1]
            init_times = [r.lod_init_time for r in result_list]
            render_times = [r.lod_render_time for r in result_list]
            
            ax4.plot(x_values, init_times, 'o-', label='LOD Init Time', linewidth=2)
            ax4.plot(x_values, render_times, 's-', label='LOD Render Time', linewidth=2)
            ax4.set_xlabel('Configuration')
            ax4.set_ylabel('Time (seconds)')
            ax4.set_title('LOD Timing Breakdown')
            ax4.set_xticks(x_values)
            ax4.set_xticklabels(x_labels, rotation=45, ha='right')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.output_dir / f"lod_performance_{config_type}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  📈 Saved plot: {plot_file}")
    
    def _create_quick_benchmark_plot(self, results: Dict[str, List[float]], colors: dict):
        """Create quick benchmark plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Quick LOD Performance Benchmark', fontsize=14, fontweight='bold')
        
        # Plot 1: Performance comparison
        point_counts = results['point_counts']
        x = np.arange(len(point_counts))
        width = 0.35
        
        ax1.bar(x - width/2, results['no_lod_times'], width, label='No LOD', color=colors['no_lod'], alpha=0.7)
        ax1.bar(x + width/2, results['lod_times'], width, label='With LOD', color=colors['lod'], alpha=0.7)
        
        ax1.set_xlabel('Point Cloud Size')
        ax1.set_ylabel('Rendering Time (seconds)')
        ax1.set_title('LOD Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{pc:,}' for pc in point_counts])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add speedup annotations
        for i, speedup in enumerate(results['speedups']):
            ax1.text(i, max(results['no_lod_times'][i], results['lod_times'][i]) + 0.1,
                    f'{speedup:.1f}x', ha='center', fontweight='bold')
        
        # Plot 2: Point reduction vs speedup
        ax2.scatter(results['point_reductions'], results['speedups'], 
                   c=results['point_counts'], cmap='viridis', s=100, alpha=0.7)
        
        ax2.set_xlabel('Point Reduction (%)')
        ax2.set_ylabel('Speedup Ratio')
        ax2.set_title('LOD Efficiency')
        ax2.grid(True, alpha=0.3)
        
        # Add point count labels
        for i, (x, y, pc) in enumerate(zip(results['point_reductions'], results['speedups'], point_counts)):
            ax2.annotate(f'{pc//1000}K', (x, y), xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / 'quick_lod_benchmark.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Saved quick benchmark plot: {plot_file}")
    
    def _create_distance_analysis_plot(self, results: Dict[str, List[float]], colors: dict):
        """Create distance analysis plot."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('LOD Performance vs Camera Distance Analysis', fontsize=16, fontweight='bold')
        
        # Convert to numpy arrays for easier manipulation
        point_counts = np.array(results['point_counts'])
        distances = np.array(results['camera_distances'])
        speedups = np.array(results['speedup_ratios'])
        reductions = np.array([results['final_point_counts'][i] / results['point_counts'][i] * 100 
                              for i in range(len(results['point_counts']))])
        
        # Get unique values
        unique_points = np.unique(point_counts)
        unique_distances = np.unique(distances)
        
        # Plot 1: Speedup heatmap
        ax1 = axes[0, 0]
        speedup_matrix = np.zeros((len(unique_points), len(unique_distances)))
        for i, pc in enumerate(unique_points):
            for j, dist in enumerate(unique_distances):
                mask = (point_counts == pc) & (distances == dist)
                if np.any(mask):
                    speedup_matrix[i, j] = speedups[mask][0]
        
        im1 = ax1.imshow(speedup_matrix, cmap='RdYlGn', aspect='auto')
        ax1.set_xlabel('Camera Distance')
        ax1.set_ylabel('Point Count')
        ax1.set_title('Speedup Ratio Heatmap')
        ax1.set_xticks(range(len(unique_distances)))
        ax1.set_xticklabels([f'{d:.1f}' for d in unique_distances])
        ax1.set_yticks(range(len(unique_points)))
        ax1.set_yticklabels([f'{int(pc/1000)}K' for pc in unique_points])
        plt.colorbar(im1, ax=ax1, label='Speedup Ratio')
        
        # Plot 2: Distance effect on largest point cloud
        ax2 = axes[0, 1]
        largest_pc = unique_points[-1]
        mask = point_counts == largest_pc
        dist_subset = distances[mask]
        speedup_subset = speedups[mask]
        
        ax2.plot(dist_subset, speedup_subset, 'o-', linewidth=2, markersize=8, color=colors['lod'])
        ax2.set_xlabel('Camera Distance')
        ax2.set_ylabel('Speedup Ratio')
        ax2.set_title(f'Distance Effect on {int(largest_pc/1000)}K Points')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        
        # Plot 3: Point count effect at far distance
        ax3 = axes[1, 0]
        far_distance = unique_distances[-1]
        mask = distances == far_distance
        pc_subset = point_counts[mask]
        speedup_subset = speedups[mask]
        
        ax3.plot(pc_subset, speedup_subset, 's-', linewidth=2, markersize=8, color=colors['lod'])
        ax3.set_xlabel('Point Count')
        ax3.set_ylabel('Speedup Ratio')
        ax3.set_title(f'Point Count Effect at Distance {far_distance:.1f}')
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        
        # Plot 4: Overall efficiency scatter
        ax4 = axes[1, 1]
        reduction_pct = [(results['point_counts'][i] - results['final_point_counts'][i]) / 
                        results['point_counts'][i] * 100 for i in range(len(results['point_counts']))]
        
        scatter = ax4.scatter(reduction_pct, speedups, c=point_counts, cmap='viridis', 
                             s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax4.set_xlabel('Point Reduction (%)')
        ax4.set_ylabel('Speedup Ratio')
        ax4.set_title('Overall LOD Efficiency')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        plt.colorbar(scatter, ax=ax4, label='Point Count')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / 'lod_distance_analysis.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📈 Saved distance analysis plot: {plot_file}")
    
    def save_results(self, comprehensive_results: Dict[str, List[BenchmarkResult]] = None,
                    quick_results: Dict[str, List[float]] = None,
                    distance_results: Dict[str, List[float]] = None,
                    real_data_results: Dict[str, List[RealDataBenchmarkResult]] = None):
        """Save all benchmark results to JSON files."""
        print("💾 Saving benchmark results...")
        
        if comprehensive_results:
            # Convert comprehensive results to serializable format
            serializable_results = {}
            for config_type, result_list in comprehensive_results.items():
                serializable_results[config_type] = [
                    {
                        'config': asdict(result.config),
                        **{k: v for k, v in asdict(result).items() if k != 'config'}
                    }
                    for result in result_list
                ]
            
            output_file = self.output_dir / "comprehensive_benchmark_results.json"
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            print(f"  📊 Comprehensive results: {output_file}")
        
        if quick_results:
            output_file = self.output_dir / "quick_benchmark_results.json"
            with open(output_file, 'w') as f:
                json.dump(quick_results, f, indent=2)
            print(f"  ⚡ Quick results: {output_file}")
        
        if distance_results:
            output_file = self.output_dir / "distance_benchmark_results.json"
            with open(output_file, 'w') as f:
                json.dump(distance_results, f, indent=2)
            print(f"  📍 Distance results: {output_file}")
        
        if real_data_results:
            # Convert real data results to serializable format
            serializable_real_data = {}
            for dataset_name, result_list in real_data_results.items():
                serializable_real_data[dataset_name] = [asdict(result) for result in result_list]
            
            output_file = self.output_dir / "real_data_benchmark_results.json"
            with open(output_file, 'w') as f:
                json.dump(serializable_real_data, f, indent=2)
            print(f"  🏗️ Real data results: {output_file}")
    
    def create_summary_report(self, comprehensive_results: Dict[str, List[BenchmarkResult]] = None,
                             quick_results: Dict[str, List[float]] = None,
                             distance_results: Dict[str, List[float]] = None,
                             real_data_results: Dict[str, List[RealDataBenchmarkResult]] = None):
        """Create a comprehensive summary report."""
        print("📋 Creating summary report...")
        
        report_lines = []
        report_lines.append("# Consolidated LOD Performance Benchmark Report")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        overall_speedups = []
        overall_reductions = []
        
        # Comprehensive results analysis
        if comprehensive_results:
            report_lines.append("## Comprehensive Benchmark Results")
            report_lines.append("")
            
            for config_type, result_list in comprehensive_results.items():
                report_lines.append(f"### {config_type.replace('_', ' ').title()} Analysis")
                report_lines.append("")
                
                speedups = [r.speedup_ratio for r in result_list]
                reductions = [r.point_reduction_pct for r in result_list]
                
                overall_speedups.extend(speedups)
                overall_reductions.extend(reductions)
                
                avg_speedup = np.mean(speedups)
                max_speedup = max(speedups)
                avg_reduction = np.mean(reductions)
                max_reduction = max(reductions)
                
                report_lines.append(f"- **Average speedup**: {avg_speedup:.2f}x")
                report_lines.append(f"- **Best speedup**: {max_speedup:.2f}x")
                report_lines.append(f"- **Average point reduction**: {avg_reduction:.1f}%")
                report_lines.append(f"- **Max point reduction**: {max_reduction:.1f}%")
                report_lines.append("")
                
                # Find best and worst cases
                best_idx = speedups.index(max_speedup)
                worst_idx = speedups.index(min(speedups))
                
                report_lines.append(f"**Best case**: {result_list[best_idx].config.name} ({max_speedup:.2f}x speedup)")
                report_lines.append(f"**Worst case**: {result_list[worst_idx].config.name} ({min(speedups):.2f}x speedup)")
                report_lines.append("")
        
        # Quick results analysis
        if quick_results:
            report_lines.append("## Quick Benchmark Results")
            report_lines.append("")
            
            avg_speedup = np.mean(quick_results['speedups'])
            max_speedup = max(quick_results['speedups'])
            avg_reduction = np.mean(quick_results['point_reductions'])
            
            overall_speedups.extend(quick_results['speedups'])
            overall_reductions.extend(quick_results['point_reductions'])
            
            report_lines.append(f"- **Average speedup**: {avg_speedup:.2f}x")
            report_lines.append(f"- **Best speedup**: {max_speedup:.2f}x")
            report_lines.append(f"- **Average point reduction**: {avg_reduction:.1f}%")
            report_lines.append("")
        
        # Distance results analysis
        if distance_results:
            report_lines.append("## Distance Analysis Results")
            report_lines.append("")
            
            avg_speedup = np.mean(distance_results['speedup_ratios'])
            max_speedup = max(distance_results['speedup_ratios'])
            
            overall_speedups.extend(distance_results['speedup_ratios'])
            
            report_lines.append(f"- **Average speedup across all distances**: {avg_speedup:.2f}x")
            report_lines.append(f"- **Best speedup**: {max_speedup:.2f}x")
            report_lines.append("")
        
        # Real data results analysis
        if real_data_results:
            report_lines.append("## Real Dataset Benchmark Results")
            report_lines.append("")
            
            for dataset_name, result_list in real_data_results.items():
                report_lines.append(f"### {dataset_name.upper()} Dataset")
                report_lines.append("")
                
                speedups = [r.speedup_ratio for r in result_list]
                reductions = [r.avg_point_reduction_pct for r in result_list]
                
                overall_speedups.extend(speedups)
                overall_reductions.extend(reductions)
                
                avg_speedup = np.mean(speedups)
                max_speedup = max(speedups)
                avg_reduction = np.mean(reductions)
                
                report_lines.append(f"- **Average speedup**: {avg_speedup:.2f}x")
                report_lines.append(f"- **Best speedup**: {max_speedup:.2f}x")
                report_lines.append(f"- **Average point reduction**: {avg_reduction:.1f}%")
                report_lines.append(f"- **Datapoints tested**: {result_list[0].num_datapoints}")
                
                # Distance group breakdown
                for result in result_list:
                    report_lines.append(f"  - **{result.camera_distance_group.capitalize()}**: {result.speedup_ratio:.2f}x speedup")
                
                report_lines.append("")
        
        # Overall summary
        if overall_speedups:
            report_lines.append("## Overall Performance Summary")
            report_lines.append("")
            report_lines.append(f"- **Overall average speedup**: {np.mean(overall_speedups):.2f}x")
            report_lines.append(f"- **Overall best speedup**: {max(overall_speedups):.2f}x")
            
            if overall_reductions:
                report_lines.append(f"- **Overall average point reduction**: {np.mean(overall_reductions):.1f}%")
            
            report_lines.append(f"- **Cases with >10% speedup**: {sum(1 for s in overall_speedups if s > 1.1)}/{len(overall_speedups)}")
            report_lines.append("")
            
            # Recommendations
            report_lines.append("## Recommendations")
            report_lines.append("")
            if np.mean(overall_speedups) > 1.5:
                report_lines.append("✅ **LOD system provides significant performance benefits**")
            elif np.mean(overall_speedups) > 1.1:
                report_lines.append("⚠️ **LOD system provides moderate performance benefits**")
            else:
                report_lines.append("❌ **LOD system may need optimization**")
            
            if overall_reductions and np.mean(overall_reductions) > 30:
                report_lines.append("✅ **LOD effectively reduces point cloud complexity**")
            elif overall_reductions:
                report_lines.append("⚠️ **LOD point reduction could be more aggressive**")
            
            report_lines.append("")
        
        # Save report
        report_file = self.output_dir / "consolidated_benchmark_report.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📄 Summary report saved: {report_file}")
    
    def print_benchmark_summary(self, results: Dict[str, List[float]]):
        """Print a comprehensive summary of benchmark results."""
        print("\n" + "="*80)
        print("🏁 BENCHMARK SUMMARY")
        print("="*80)
        
        if 'point_counts' in results:
            print(f"{'Points':<10} {'Distance':<8} {'No LOD (s)':<12} {'LOD (s)':<10} {'LOD Lvl':<8} {'Final Pts':<12} {'Speedup':<8} {'Init OH (s)':<10}")
            print("-"*80)
            
            total_cases = len(results['point_counts'])
            significant_speedups = 0
            total_speedup = 0
            
            for i in range(total_cases):
                points = results['point_counts'][i]
                distance = results.get('camera_distances', [0.0] * total_cases)[i]
                no_lod = results.get('lod_disabled_times', results.get('no_lod_times', [0.0] * total_cases))[i]
                lod_time = results.get('lod_enabled_times', results.get('lod_times', [0.0] * total_cases))[i]
                lod_level = results.get('lod_levels', [0] * total_cases)[i]
                final_pts = results.get('final_point_counts', results.get('point_counts', [0] * total_cases))[i]
                speedup = results.get('speedup_ratios', results.get('speedups', [1.0] * total_cases))[i]
                init_oh = results.get('initialization_overhead', [0.0] * total_cases)[i]
                
                print(f"{points:<10,} {distance:<8.1f} {no_lod:<12.4f} {lod_time:<10.4f} {lod_level:<8} {final_pts:<12,} {speedup:<8.2f} {init_oh:<10.4f}")
                
                total_speedup += speedup
                if speedup > 1.1:  # At least 10% improvement
                    significant_speedups += 1
            
            avg_speedup = total_speedup / total_cases if total_cases > 0 else 1.0
            
            print("-"*80)
            print(f"📈 PERFORMANCE ANALYSIS:")
            print(f"   • Average speedup: {avg_speedup:.2f}x")
            print(f"   • Cases with >10% speedup: {significant_speedups}/{total_cases}")
            print(f"   • Best speedup: {max(results.get('speedup_ratios', results.get('speedups', [1.0]))):.2f}x")
            print(f"   • Worst case: {min(results.get('speedup_ratios', results.get('speedups', [1.0]))):.2f}x")
        
        print("="*80)
    
    def run_all_benchmarks(self, mode: str = "comprehensive", data_root: str = None):
        """Run all benchmark modes based on the specified mode.
        
        Args:
            mode: "quick", "comprehensive", "distance", "real_data", or "all"
            data_root: Path to dataset root directory (required for real_data mode)
        """
        print(f"🚀 Starting consolidated LOD benchmark in '{mode}' mode...")
        
        comprehensive_results = None
        quick_results = None
        distance_results = None
        real_data_results = None
        
        if mode in ["quick", "all"]:
            print("\n⚡ Running quick benchmark...")
            quick_results = self.run_quick_benchmark()
            
        if mode in ["comprehensive", "all"]:
            print("\n📊 Running comprehensive benchmark suite...")
            comprehensive_results = self.run_comprehensive_benchmark_suite()
            
        if mode in ["distance", "all"]:
            print("\n📍 Running detailed distance benchmark...")
            distance_results = self.run_detailed_distance_benchmark()
        
        if mode in ["real_data", "all"]:
            if data_root is None:
                print("\n⚠️ data_root required for real_data mode, skipping...")
            else:
                print("\n🏗️ Running real data benchmark...")
                real_data_results = self.run_real_data_benchmark(data_root, num_samples=5, num_poses_per_distance=3)
        
        # Create visualizations
        self.create_comprehensive_plots(comprehensive_results, quick_results, distance_results, real_data_results)
        
        # Save results
        self.save_results(comprehensive_results, quick_results, distance_results, real_data_results)
        
        # Create summary report
        self.create_summary_report(comprehensive_results, quick_results, distance_results, real_data_results)
        
        # Print summaries
        if distance_results:
            self.print_benchmark_summary(distance_results)
        elif quick_results:
            self.print_benchmark_summary(quick_results)
        
        print(f"\n✅ Consolidated benchmark complete! Results saved in: {self.output_dir}")
        print(f"📊 Check the PNG files for visual performance analysis")
        print(f"📄 Check consolidated_benchmark_report.md for summary")


def main():
    """Run the complete consolidated LOD benchmark suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Consolidated LOD Benchmark Suite")
    parser.add_argument("--mode", choices=["quick", "comprehensive", "distance", "real_data", "all"], 
                       default="comprehensive",
                       help="Benchmark mode to run (default: comprehensive)")
    parser.add_argument("--data-root", type=str, default=None,
                       help="Path to dataset root directory (required for real_data mode)")
    
    args = parser.parse_args()
    
    print("🚀 Starting consolidated LOD benchmark with visual reports...")
    
    benchmark = ConsolidatedLODBenchmark()
    benchmark.run_all_benchmarks(mode=args.mode, data_root=args.data_root)


if __name__ == "__main__":
    main()