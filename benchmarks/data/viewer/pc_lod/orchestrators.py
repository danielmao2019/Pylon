"""Benchmark orchestrators for coordinating synthetic and real data benchmarks."""

from typing import Dict, List, Any
import torch
import numpy as np
from pathlib import Path

from .data_types import PointCloudSample, BenchmarkStats
from .streamers import SyntheticPointCloudStreamer, RealDataPointCloudStreamer
from .camera_poses import CameraPoseSampler
from .benchmark_runner import LODBenchmarkRunner


class SyntheticBenchmarkOrchestrator:
    """Orchestrates synthetic data benchmarks."""
    
    def __init__(self, output_dir: str = "benchmarks/data/viewer/pc_lod/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.point_cloud_streamer = SyntheticPointCloudStreamer()
        self.camera_pose_sampler = CameraPoseSampler()
        self.benchmark_runner = LODBenchmarkRunner()
    
    def run_comprehensive_benchmark(self, num_configs: int = None) -> Dict[str, Dict[str, List[BenchmarkStats]]]:
        """Run comprehensive synthetic benchmark across all configurations."""
        print("📊 Running comprehensive synthetic benchmark...")
        
        # Group results by dataset and distance group
        results_by_dataset = {}
        
        for sample in self.point_cloud_streamer.stream_point_clouds(num_configs):
            print(f"  🔍 Testing {sample.name}...")
            
            # Generate camera poses for all distance groups (1 pose per distance for speed)
            camera_poses = self.camera_pose_sampler.generate_poses_for_point_cloud(sample.points, num_poses_per_distance=1)
            
            dataset_name = sample.metadata['dataset']  # e.g., 'small', 'medium', 'large', 'xlarge'
            
            if dataset_name not in results_by_dataset:
                results_by_dataset[dataset_name] = {'close': [], 'medium': [], 'far': []}
            
            # Run benchmark for each camera pose
            for camera_pose in camera_poses:
                stats = self.benchmark_runner.benchmark_single_pose(sample, camera_pose)
                results_by_dataset[dataset_name][camera_pose.distance_group].append(stats)
                
            # Print summary for this sample (use far pose for feedback)
            far_poses = [pose for pose in camera_poses if pose.distance_group == 'far']
            if far_poses:
                far_stats = results_by_dataset[dataset_name]['far'][-len(far_poses):]  # Latest far results
                avg_speedup = np.mean([s.speedup_ratio for s in far_stats])
                avg_reduction = np.mean([s.point_reduction_pct for s in far_stats])
                print(f"    📈 {avg_speedup:.2f}x speedup, {avg_reduction:.1f}% reduction")
        
        return results_by_dataset
    
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
    
    def run_real_data_benchmark(self, num_samples: int = 10, 
                               num_poses_per_distance: int = 3) -> Dict[str, Dict[str, List[BenchmarkStats]]]:
        """Run LOD benchmark on real datasets.
        
        Args:
            num_samples: Number of datapoints to sample from each dataset
            num_poses_per_distance: Number of camera poses per distance group
            
        Returns:
            Nested dictionary: {dataset_name: {distance_group: [BenchmarkStats]}}
        """
        print("🏗️ Running real data benchmark...")
        
        # Initialize real data streamer
        real_data_streamer = RealDataPointCloudStreamer()
        
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