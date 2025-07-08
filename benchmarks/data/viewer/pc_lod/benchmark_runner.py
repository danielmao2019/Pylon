"""Core benchmark execution for LOD performance testing."""

from typing import Any
import sys
import os
import time
import gc
import numpy as np

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, project_root)

from data.viewer.utils.point_cloud import create_point_cloud_figure
# LOD imports not needed since point_cloud.py handles it internally

from .data_types import PointCloudSample, CameraPose, BenchmarkStats


class LODBenchmarkRunner:
    """Runs individual LOD benchmarks for point cloud and camera pose pairs."""
    
    def __init__(self, num_runs: int = 3):
        self.num_runs = num_runs
        
        # Clear LOD cache for discrete LOD
        discrete_lod = DiscreteLOD()
        discrete_lod.clear_cache()
    
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
                if "LOD" in title and "(LOD " in title and ":" in title:
                    title_parts = title.split("(LOD ")
                    if len(title_parts) > 1:
                        lod_part = title_parts[1].split(":")[0].strip()
                        lod_info['level'] = int(lod_part)
                        
                        colon_parts = title.split(": ")
                        if len(colon_parts) > 1:
                            points_part = colon_parts[1].split("/")[0]
                            lod_info['final_points'] = int(points_part.replace(",", ""))
        
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