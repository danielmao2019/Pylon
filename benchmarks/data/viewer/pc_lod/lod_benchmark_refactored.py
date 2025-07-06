#!/usr/bin/env python3
"""Refactored LOD benchmark with modular architecture."""

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
from typing import Dict, List, Tuple, Any, Iterator, Optional
from pathlib import Path
import json
import gc
import random
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

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
class PointCloudSample:
    """A single point cloud sample with metadata."""
    name: str
    points: torch.Tensor
    colors: torch.Tensor
    source: str
    metadata: Dict[str, Any] = None


@dataclass
class CameraPose:
    """A camera pose configuration."""
    camera_state: Dict[str, Any]
    distance_group: str  # 'close', 'medium', 'far'
    distance_value: float
    pose_id: int


@dataclass
class BenchmarkStats:
    """Statistics from a single benchmark run."""
    point_cloud_name: str
    camera_pose_info: str
    original_points: int
    final_points: int
    point_reduction_pct: float
    lod_level: int
    no_lod_time: float
    lod_time: float
    speedup_ratio: float
    num_runs: int


# ============================================================================
# 1. POINT CLOUD STREAMER
# ============================================================================

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
        if not REAL_DATA_AVAILABLE:
            return []
            
        try:
            dataset = Urb3DCDDataset(
                data_root=self.data_root,
                split='train',
                patched=False,
                deterministic_seed=self.seed
            )
            
            total_samples = min(len(dataset), num_samples * 5)
            sample_indices = random.sample(range(total_samples), min(total_samples, num_samples))
            
            samples = []
            for idx in sample_indices:
                try:
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
                        
                except Exception as e:
                    print(f"Failed to load URB3DCD sample {idx}: {e}")
                    continue
            
            return samples[:num_samples * 2]
            
        except Exception as e:
            print(f"Failed to initialize URB3DCD dataset: {e}")
            return []
    
    def _load_slpccd_samples(self, num_samples: int) -> List[PointCloudSample]:
        """Load samples from SLPCCD dataset."""
        if not REAL_DATA_AVAILABLE:
            return []
            
        try:
            dataset = SLPCCDDataset(
                data_root=self.data_root,
                split='train',
                use_hierarchy=False,
                deterministic_seed=self.seed
            )
            
            total_samples = min(len(dataset), num_samples * 5)
            sample_indices = random.sample(range(total_samples), min(total_samples, num_samples))
            
            samples = []
            for idx in sample_indices:
                try:
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
                        
                except Exception as e:
                    print(f"Failed to load SLPCCD sample {idx}: {e}")
                    continue
            
            return samples[:num_samples * 2]
            
        except Exception as e:
            print(f"Failed to initialize SLPCCD dataset: {e}")
            return []
    
    def _load_kitti_samples(self, num_samples: int) -> List[PointCloudSample]:
        """Load samples from KITTI dataset."""
        if not REAL_DATA_AVAILABLE:
            return []
            
        try:
            dataset = KITTIDataset(
                data_root=self.data_root,
                split='train',
                deterministic_seed=self.seed
            )
            
            total_samples = min(len(dataset), num_samples * 5)
            sample_indices = random.sample(range(total_samples), min(total_samples, num_samples))
            
            samples = []
            for idx in sample_indices:
                try:
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
                        
                except Exception as e:
                    print(f"Failed to load KITTI sample {idx}: {e}")
                    continue
            
            return samples[:num_samples * 2]
            
        except Exception as e:
            print(f"Failed to initialize KITTI dataset: {e}")
            return []
    
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


# ============================================================================
# 2. CAMERA POSE SAMPLER  
# ============================================================================

class CameraPoseSampler:
    """Generates camera poses for point cloud benchmarking."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        
    def generate_poses_for_point_cloud(self, point_cloud: torch.Tensor, 
                                     num_poses_per_distance: int = 3) -> List[CameraPose]:
        """Generate camera poses at different distances from point cloud.
        
        Args:
            point_cloud: Point cloud tensor of shape (N, 3)
            num_poses_per_distance: Number of camera poses to generate per distance group
            
        Returns:
            List of camera poses across all distance groups
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
        
        all_poses = []
        
        for group_name, base_distance in distance_groups.items():
            for i in range(num_poses_per_distance):
                # Generate camera position facing toward the point cloud
                theta = self.rng.uniform(0, 2 * np.pi)  # Azimuth angle
                phi = self.rng.uniform(np.pi/6, np.pi/3)  # Elevation angle (avoid top-down)
                
                # Convert spherical to cartesian coordinates
                x = base_distance * np.sin(phi) * np.cos(theta)
                y = base_distance * np.sin(phi) * np.sin(theta) 
                z = base_distance * np.cos(phi)
                
                # Add some translation to the side (perpendicular to view direction)
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
                
                pose = CameraPose(
                    camera_state=camera_state,
                    distance_group=group_name,
                    distance_value=base_distance,
                    pose_id=i
                )
                
                all_poses.append(pose)
        
        return all_poses


# ============================================================================
# 3. BENCHMARK RUNNER
# ============================================================================

class LODBenchmarkRunner:
    """Runs individual LOD benchmarks for point cloud and camera pose pairs."""
    
    def __init__(self, num_runs: int = 3):
        self.num_runs = num_runs
        
        # Clear LOD cache
        lod_manager = get_lod_manager()
        lod_manager.clear_cache()
    
    def benchmark_single_pose(self, point_cloud_sample: PointCloudSample, 
                             camera_pose: CameraPose) -> BenchmarkStats:
        """Run LOD benchmark for a single point cloud and camera pose.
        
        Args:
            point_cloud_sample: Point cloud sample to benchmark
            camera_pose: Camera pose configuration
            
        Returns:
            Benchmark statistics
        """
        points = point_cloud_sample.points
        colors = point_cloud_sample.colors
        
        # Benchmark WITHOUT LOD
        no_lod_times = []
        for run in range(self.num_runs):
            gc.collect()  # Clean memory
            
            start_time = time.perf_counter()
            fig_no_lod = create_point_cloud_figure(
                points=points,
                colors=colors,
                title=f"No LOD {run}",
                camera_state=camera_pose.camera_state,
                lod_enabled=False
            )
            end_time = time.perf_counter()
            
            no_lod_times.append(end_time - start_time)
        
        # Benchmark WITH LOD
        lod_times = []
        lod_info = {'level': 0, 'final_points': len(points)}
        
        for run in range(self.num_runs):
            gc.collect()  # Clean memory
            
            start_time = time.perf_counter()
            fig_lod = create_point_cloud_figure(
                points=points,
                colors=colors,
                title=f"LOD {run}",
                camera_state=camera_pose.camera_state,
                lod_enabled=True,
                point_cloud_id=f"{point_cloud_sample.name}_{camera_pose.distance_group}_{camera_pose.pose_id}"
            )
            end_time = time.perf_counter()
            
            lod_times.append(end_time - start_time)
            
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
        
        # Calculate statistics
        avg_no_lod_time = np.mean(no_lod_times)
        avg_lod_time = np.mean(lod_times)
        
        original_points = len(points)
        final_points = lod_info['final_points']
        point_reduction_pct = (original_points - final_points) / original_points * 100
        speedup_ratio = avg_no_lod_time / avg_lod_time if avg_lod_time > 0 else 1.0
        
        return BenchmarkStats(
            point_cloud_name=point_cloud_sample.name,
            camera_pose_info=f"{camera_pose.distance_group}_{camera_pose.pose_id}",
            original_points=original_points,
            final_points=final_points,
            point_reduction_pct=point_reduction_pct,
            lod_level=lod_info['level'],
            no_lod_time=avg_no_lod_time,
            lod_time=avg_lod_time,
            speedup_ratio=speedup_ratio,
            num_runs=self.num_runs
        )


# ============================================================================
# 4. BENCHMARK ORCHESTRATORS
# ============================================================================

class SyntheticBenchmarkOrchestrator:
    """Orchestrates synthetic data benchmarks."""
    
    def __init__(self, output_dir: str = "benchmarks/data/viewer/pc_lod/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.point_cloud_streamer = SyntheticPointCloudStreamer()
        self.camera_pose_sampler = CameraPoseSampler()
        self.benchmark_runner = LODBenchmarkRunner()
    
    def run_comprehensive_benchmark(self) -> Dict[str, List[BenchmarkStats]]:
        """Run comprehensive synthetic benchmark across all configurations."""
        print("📊 Running comprehensive synthetic benchmark...")
        
        # Group results by configuration category
        results_by_category = {}
        
        for sample in self.point_cloud_streamer.stream_point_clouds():
            print(f"  🔍 Testing {sample.name}...")
            
            # Generate camera poses (use far distance for comprehensive test)
            camera_poses = self.camera_pose_sampler.generate_poses_for_point_cloud(sample.points, num_poses_per_distance=1)
            far_poses = [pose for pose in camera_poses if pose.distance_group == 'far']
            
            if far_poses:
                # Use the first far pose for comprehensive benchmark
                camera_pose = far_poses[0]
                
                # Run benchmark
                stats = self.benchmark_runner.benchmark_single_pose(sample, camera_pose)
                
                # Group by category
                category = sample.metadata['category']
                if category not in results_by_category:
                    results_by_category[category] = []
                results_by_category[category].append(stats)
                
                print(f"    📈 {stats.speedup_ratio:.2f}x speedup, {stats.point_reduction_pct:.1f}% reduction")
        
        return results_by_category
    
    def run_quick_benchmark(self) -> List[BenchmarkStats]:
        """Run quick synthetic benchmark on selected configurations."""
        print("⚡ Running quick synthetic benchmark...")
        
        # Use specific configurations for quick test
        quick_configs = [
            {'num_points': 10000, 'spatial_size': 10.0, 'shape': 'sphere', 'density_factor': 1.0, 'name': '10K', 'category': 'quick'},
            {'num_points': 50000, 'spatial_size': 10.0, 'shape': 'sphere', 'density_factor': 1.0, 'name': '50K', 'category': 'quick'},
            {'num_points': 100000, 'spatial_size': 10.0, 'shape': 'sphere', 'density_factor': 1.0, 'name': '100K', 'category': 'quick'}
        ]
        
        results = []
        
        for config in quick_configs:
            print(f"  📊 Testing {config['name']} points...")
            
            # Generate point cloud
            if config['shape'] == 'sphere':
                points = self.point_cloud_streamer._generate_sphere(
                    config['num_points'], config['spatial_size'], config['density_factor']
                )
            colors = torch.rand(config['num_points'], 3)
            
            sample = PointCloudSample(
                name=config['name'],
                points=points,
                colors=colors,
                source='synthetic',
                metadata=config
            )
            
            # Generate far camera pose
            camera_poses = self.camera_pose_sampler.generate_poses_for_point_cloud(sample.points, num_poses_per_distance=1)
            far_poses = [pose for pose in camera_poses if pose.distance_group == 'far']
            
            if far_poses:
                stats = self.benchmark_runner.benchmark_single_pose(sample, far_poses[0])
                results.append(stats)
                
                print(f"    📈 {stats.speedup_ratio:.2f}x speedup, {stats.point_reduction_pct:.1f}% reduction")
        
        return results


class RealDataBenchmarkOrchestrator:
    """Orchestrates real data benchmarks."""
    
    def __init__(self, output_dir: str = "benchmarks/data/viewer/pc_lod/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.camera_pose_sampler = CameraPoseSampler()
        self.benchmark_runner = LODBenchmarkRunner()
    
    def run_real_data_benchmark(self, data_root: str, num_samples: int = 10, 
                               num_poses_per_distance: int = 3) -> Dict[str, Dict[str, List[BenchmarkStats]]]:
        """Run LOD benchmark on real datasets.
        
        Args:
            data_root: Path to dataset root directory
            num_samples: Number of datapoints to sample from each dataset
            num_poses_per_distance: Number of camera poses per distance group
            
        Returns:
            Nested dictionary: {dataset_name: {distance_group: [BenchmarkStats]}}
        """
        if not REAL_DATA_AVAILABLE:
            print("⚠️ Real datasets not available, skipping real data benchmark")
            return {}
            
        print("🏗️ Running real data benchmark...")
        
        # Initialize real data streamer
        real_data_streamer = RealDataPointCloudStreamer(data_root)
        
        # Group results by dataset and distance group
        all_results = {}
        
        current_dataset = None
        dataset_samples = []
        
        # Process samples as they stream
        for sample in real_data_streamer.stream_point_clouds(num_samples):
            # Group samples by dataset
            if current_dataset != sample.source:
                # Process previous dataset if exists
                if current_dataset and dataset_samples:
                    dataset_results = self._process_dataset_samples(
                        current_dataset, dataset_samples, num_poses_per_distance
                    )
                    all_results[current_dataset] = dataset_results
                
                # Start new dataset
                current_dataset = sample.source
                dataset_samples = []
                print(f"\n📊 Processing {current_dataset.upper()} dataset...")
            
            dataset_samples.append(sample)
        
        # Process final dataset
        if current_dataset and dataset_samples:
            dataset_results = self._process_dataset_samples(
                current_dataset, dataset_samples, num_poses_per_distance
            )
            all_results[current_dataset] = dataset_results
        
        return all_results
    
    def _process_dataset_samples(self, dataset_name: str, samples: List[PointCloudSample], 
                               num_poses_per_distance: int) -> Dict[str, List[BenchmarkStats]]:
        """Process all samples from a single dataset."""
        # Group results by distance group
        distance_results = {'close': [], 'medium': [], 'far': []}
        
        for sample in samples:
            print(f"  🔍 Processing {sample.name}...")
            
            # Generate camera poses for this point cloud
            camera_poses = self.camera_pose_sampler.generate_poses_for_point_cloud(
                sample.points, num_poses_per_distance
            )
            
            # Benchmark each camera pose
            for camera_pose in camera_poses:
                stats = self.benchmark_runner.benchmark_single_pose(sample, camera_pose)
                distance_results[camera_pose.distance_group].append(stats)
        
        # Print summary for this dataset
        for distance_group, stats_list in distance_results.items():
            if stats_list:
                avg_speedup = np.mean([s.speedup_ratio for s in stats_list])
                avg_reduction = np.mean([s.point_reduction_pct for s in stats_list])
                print(f"    📈 {distance_group.capitalize()}: {avg_speedup:.2f}x speedup, {avg_reduction:.1f}% reduction")
        
        return distance_results


# ============================================================================
# 5. REPORT GENERATOR
# ============================================================================

class BenchmarkReportGenerator:
    """Generates benchmark reports and visualizations."""
    
    def __init__(self, output_dir: str = "benchmarks/data/viewer/pc_lod/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_synthetic_plots(self, results_by_category: Dict[str, List[BenchmarkStats]]):
        """Create plots for synthetic benchmark results."""
        print("📊 Creating synthetic benchmark plots...")
        
        plt.style.use('seaborn-v0_8')
        colors = {'lod': '#2E86C1', 'no_lod': '#E74C3C'}
        
        for category, stats_list in results_by_category.items():
            if not stats_list:
                continue
                
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'LOD Performance Analysis: {category.replace("_", " ").title()}', 
                        fontsize=16, fontweight='bold')
            
            # Extract data
            names = [s.point_cloud_name for s in stats_list]
            no_lod_times = [s.no_lod_time for s in stats_list]
            lod_times = [s.lod_time for s in stats_list]
            speedups = [s.speedup_ratio for s in stats_list]
            reductions = [s.point_reduction_pct for s in stats_list]
            
            # Plot 1: Time comparison
            ax1 = axes[0, 0]
            x = np.arange(len(names))
            width = 0.35
            
            ax1.bar(x - width/2, no_lod_times, width, label='No LOD', color=colors['no_lod'], alpha=0.8)
            ax1.bar(x + width/2, lod_times, width, label='With LOD', color=colors['lod'], alpha=0.8)
            
            ax1.set_xlabel('Configuration')
            ax1.set_ylabel('Rendering Time (seconds)')
            ax1.set_title('Rendering Time Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(names, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Speedup ratios
            ax2 = axes[0, 1]
            bars = ax2.bar(x, speedups, color=[colors['lod'] if s > 1.1 else 'gray' for s in speedups], alpha=0.8)
            ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
            ax2.set_xlabel('Configuration')
            ax2.set_ylabel('Speedup Ratio')
            ax2.set_title('Performance Speedup')
            ax2.set_xticks(x)
            ax2.set_xticklabels(names, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            # Add speedup labels
            for bar, speedup in zip(bars, speedups):
                height = bar.get_height()
                ax2.annotate(f'{speedup:.1f}x',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
            
            # Plot 3: Point reduction
            ax3 = axes[1, 0]
            ax3.bar(x, reductions, color=colors['lod'], alpha=0.8)
            ax3.set_xlabel('Configuration')
            ax3.set_ylabel('Point Reduction (%)')
            ax3.set_title('Point Cloud Reduction')
            ax3.set_xticks(x)
            ax3.set_xticklabels(names, rotation=45, ha='right')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Efficiency scatter
            ax4 = axes[1, 1]
            final_points = [s.final_points for s in stats_list]
            scatter = ax4.scatter(reductions, speedups, c=final_points, cmap='viridis', 
                                s=100, alpha=0.7, edgecolors='black', linewidth=1)
            ax4.set_xlabel('Point Reduction (%)')
            ax4.set_ylabel('Speedup Ratio')
            ax4.set_title('LOD Efficiency')
            ax4.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax4, label='Final Point Count')
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.output_dir / f"synthetic_{category}_performance.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  📈 Saved {category} plot: {plot_file}")
    
    def create_real_data_plots(self, real_data_results: Dict[str, Dict[str, List[BenchmarkStats]]]):
        """Create plots for real dataset benchmark results."""
        if not real_data_results:
            return
            
        print("📊 Creating real data performance plots...")
        
        plt.style.use('seaborn-v0_8')
        colors = {'lod': '#2E86C1', 'no_lod': '#E74C3C'}
        
        # Create a figure for each dataset
        for dataset_name, distance_results in real_data_results.items():
            if not any(distance_results.values()):
                continue
                
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.suptitle(f'LOD Performance on {dataset_name.upper()} Dataset', fontsize=14, fontweight='bold')
            
            # Aggregate results by distance group
            distance_groups = []
            no_lod_times = []
            lod_times = []
            speedups = []
            
            for distance_group in ['close', 'medium', 'far']:
                stats_list = distance_results.get(distance_group, [])
                if stats_list:
                    distance_groups.append(distance_group)
                    no_lod_times.append(np.mean([s.no_lod_time for s in stats_list]))
                    lod_times.append(np.mean([s.lod_time for s in stats_list]))
                    speedups.append(np.mean([s.speedup_ratio for s in stats_list]))
            
            if not distance_groups:
                continue
            
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
            ax.set_title(f'Average Performance Across Multiple Datapoints and Camera Poses')
            ax.set_xticks(x)
            ax.set_xticklabels([g.capitalize() for g in distance_groups])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            avg_speedup = np.mean(speedups)
            all_stats = [s for stats_list in distance_results.values() for s in stats_list]
            avg_reduction = np.mean([s.point_reduction_pct for s in all_stats]) if all_stats else 0
            
            stats_text = f"Average Speedup: {avg_speedup:.2f}x\nAverage Point Reduction: {avg_reduction:.1f}%"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.output_dir / f"real_data_{dataset_name}_performance.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  📈 Saved {dataset_name} plot: {plot_file}")
    
    def save_results_json(self, synthetic_results: Dict[str, List[BenchmarkStats]] = None,
                         real_data_results: Dict[str, Dict[str, List[BenchmarkStats]]] = None,
                         quick_results: List[BenchmarkStats] = None):
        """Save benchmark results to JSON files."""
        print("💾 Saving benchmark results...")
        
        if synthetic_results:
            # Convert to serializable format
            serializable_synthetic = {}
            for category, stats_list in synthetic_results.items():
                serializable_synthetic[category] = [asdict(stats) for stats in stats_list]
            
            output_file = self.output_dir / "synthetic_benchmark_results.json"
            with open(output_file, 'w') as f:
                json.dump(serializable_synthetic, f, indent=2)
            print(f"  📊 Synthetic results: {output_file}")
        
        if real_data_results:
            # Convert to serializable format
            serializable_real = {}
            for dataset_name, distance_results in real_data_results.items():
                serializable_real[dataset_name] = {}
                for distance_group, stats_list in distance_results.items():
                    serializable_real[dataset_name][distance_group] = [asdict(stats) for stats in stats_list]
            
            output_file = self.output_dir / "real_data_benchmark_results.json"
            with open(output_file, 'w') as f:
                json.dump(serializable_real, f, indent=2)
            print(f"  🏗️ Real data results: {output_file}")
        
        if quick_results:
            serializable_quick = [asdict(stats) for stats in quick_results]
            
            output_file = self.output_dir / "quick_benchmark_results.json"
            with open(output_file, 'w') as f:
                json.dump(serializable_quick, f, indent=2)
            print(f"  ⚡ Quick results: {output_file}")
    
    def create_summary_report(self, synthetic_results: Dict[str, List[BenchmarkStats]] = None,
                             real_data_results: Dict[str, Dict[str, List[BenchmarkStats]]] = None,
                             quick_results: List[BenchmarkStats] = None):
        """Create a comprehensive summary report."""
        print("📋 Creating summary report...")
        
        report_lines = []
        report_lines.append("# Refactored LOD Performance Benchmark Report")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        overall_speedups = []
        overall_reductions = []
        
        # Quick results
        if quick_results:
            report_lines.append("## Quick Benchmark Results")
            report_lines.append("")
            
            speedups = [s.speedup_ratio for s in quick_results]
            reductions = [s.point_reduction_pct for s in quick_results]
            
            overall_speedups.extend(speedups)
            overall_reductions.extend(reductions)
            
            report_lines.append(f"- **Average speedup**: {np.mean(speedups):.2f}x")
            report_lines.append(f"- **Best speedup**: {max(speedups):.2f}x")
            report_lines.append(f"- **Average point reduction**: {np.mean(reductions):.1f}%")
            report_lines.append("")
        
        # Synthetic results
        if synthetic_results:
            report_lines.append("## Synthetic Benchmark Results")
            report_lines.append("")
            
            for category, stats_list in synthetic_results.items():
                report_lines.append(f"### {category.replace('_', ' ').title()} Analysis")
                report_lines.append("")
                
                speedups = [s.speedup_ratio for s in stats_list]
                reductions = [s.point_reduction_pct for s in stats_list]
                
                overall_speedups.extend(speedups)
                overall_reductions.extend(reductions)
                
                report_lines.append(f"- **Average speedup**: {np.mean(speedups):.2f}x")
                report_lines.append(f"- **Best speedup**: {max(speedups):.2f}x")
                report_lines.append(f"- **Average point reduction**: {np.mean(reductions):.1f}%")
                report_lines.append("")
        
        # Real data results
        if real_data_results:
            report_lines.append("## Real Dataset Benchmark Results")
            report_lines.append("")
            
            for dataset_name, distance_results in real_data_results.items():
                report_lines.append(f"### {dataset_name.upper()} Dataset")
                report_lines.append("")
                
                all_stats = [s for stats_list in distance_results.values() for s in stats_list]
                if all_stats:
                    speedups = [s.speedup_ratio for s in all_stats]
                    reductions = [s.point_reduction_pct for s in all_stats]
                    
                    overall_speedups.extend(speedups)
                    overall_reductions.extend(reductions)
                    
                    report_lines.append(f"- **Average speedup**: {np.mean(speedups):.2f}x")
                    report_lines.append(f"- **Best speedup**: {max(speedups):.2f}x")
                    report_lines.append(f"- **Average point reduction**: {np.mean(reductions):.1f}%")
                    
                    # Distance group breakdown
                    for distance_group in ['close', 'medium', 'far']:
                        stats_list = distance_results.get(distance_group, [])
                        if stats_list:
                            group_speedup = np.mean([s.speedup_ratio for s in stats_list])
                            report_lines.append(f"  - **{distance_group.capitalize()}**: {group_speedup:.2f}x speedup")
                    
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
        report_file = self.output_dir / "refactored_benchmark_report.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📄 Summary report saved: {report_file}")


# ============================================================================
# 6. MAIN ORCHESTRATOR
# ============================================================================

class RefactoredLODBenchmark:
    """Main orchestrator for the refactored LOD benchmark system."""
    
    def __init__(self, output_dir: str = "benchmarks/data/viewer/pc_lod/results"):
        self.output_dir = output_dir
        
        # Initialize components
        self.synthetic_orchestrator = SyntheticBenchmarkOrchestrator(output_dir)
        self.real_data_orchestrator = RealDataBenchmarkOrchestrator(output_dir)
        self.report_generator = BenchmarkReportGenerator(output_dir)
    
    def run_benchmark_suite(self, mode: str = "comprehensive", data_root: str = None):
        """Run benchmark suite based on specified mode.
        
        Args:
            mode: "quick", "comprehensive", "real_data", or "all"
            data_root: Path to dataset root directory (required for real_data mode)
        """
        print(f"🚀 Starting refactored LOD benchmark in '{mode}' mode...")
        
        synthetic_results = None
        real_data_results = None
        quick_results = None
        
        if mode in ["quick", "all"]:
            print("\n⚡ Running quick benchmark...")
            quick_results = self.synthetic_orchestrator.run_quick_benchmark()
            
        if mode in ["comprehensive", "all"]:
            print("\n📊 Running comprehensive synthetic benchmark...")
            synthetic_results = self.synthetic_orchestrator.run_comprehensive_benchmark()
            
        if mode in ["real_data", "all"]:
            if data_root is None:
                print("\n⚠️ data_root required for real_data mode, skipping...")
            else:
                print("\n🏗️ Running real data benchmark...")
                real_data_results = self.real_data_orchestrator.run_real_data_benchmark(
                    data_root, num_samples=5, num_poses_per_distance=3
                )
        
        # Generate reports and visualizations
        print("\n📊 Generating reports and visualizations...")
        
        if synthetic_results:
            self.report_generator.create_synthetic_plots(synthetic_results)
        
        if real_data_results:
            self.report_generator.create_real_data_plots(real_data_results)
        
        # Save results
        self.report_generator.save_results_json(synthetic_results, real_data_results, quick_results)
        
        # Create summary report
        self.report_generator.create_summary_report(synthetic_results, real_data_results, quick_results)
        
        print(f"\n✅ Refactored benchmark complete! Results saved in: {self.output_dir}")
        print(f"📊 Check the PNG files for visual performance analysis")
        print(f"📄 Check refactored_benchmark_report.md for summary")


def main():
    """Run the refactored LOD benchmark suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Refactored LOD Benchmark Suite")
    parser.add_argument("--mode", choices=["quick", "comprehensive", "real_data", "all"], 
                       default="comprehensive",
                       help="Benchmark mode to run (default: comprehensive)")
    parser.add_argument("--data-root", type=str, default=None,
                       help="Path to dataset root directory (required for real_data mode)")
    
    args = parser.parse_args()
    
    print("🚀 Starting refactored LOD benchmark with modular architecture...")
    
    benchmark = RefactoredLODBenchmark()
    benchmark.run_benchmark_suite(mode=args.mode, data_root=args.data_root)


if __name__ == "__main__":
    main()
