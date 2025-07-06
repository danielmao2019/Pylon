#!/usr/bin/env python3
"""Comprehensive benchmark for point cloud LOD rendering performance with visual reports."""

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
from dataclasses import dataclass, asdict
from data.viewer.utils.point_cloud import create_point_cloud_figure
from data.viewer.utils.camera_lod import get_lod_manager


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


class LODBenchmark:
    """Comprehensive LOD performance benchmark."""
    
    def __init__(self, output_dir: str = "benchmarks/data/viewer/pc_lod/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear LOD cache before benchmarking
        lod_manager = get_lod_manager()
        lod_manager.clear_cache()
        
        # Benchmark configuration
        self.num_runs = 3
        self.camera_distances = [0.1, 2.0, 10.0]  # Close, medium, far
        
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
            render_speedup=render_speedup
        )
    
    def run_benchmark_suite(self) -> Dict[str, List[BenchmarkResult]]:
        """Run complete benchmark suite."""
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
        
        # Save raw results
        self.save_results(all_results)
        
        return all_results
    
    def save_results(self, results: Dict[str, List[BenchmarkResult]]):
        """Save benchmark results to JSON file."""
        # Convert to serializable format
        serializable_results = {}
        for config_type, result_list in results.items():
            serializable_results[config_type] = [
                {
                    'config': asdict(result.config),
                    **{k: v for k, v in asdict(result).items() if k != 'config'}
                }
                for result in result_list
            ]
        
        output_file = self.output_dir / "benchmark_results.json"
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Results saved to {output_file}")
    
    def create_performance_plots(self, results: Dict[str, List[BenchmarkResult]]):
        """Create comprehensive performance visualization plots."""
        print("📊 Creating performance visualization plots...")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        colors = {
            'lod': '#2E86C1',      # Blue for LOD
            'no_lod': '#E74C3C'    # Red for no LOD
        }
        line_styles = {
            'init': '--',           # Dashed for init time
            'render': '-',          # Solid for render time
            'total': '-.'           # Dash-dot for total time
        }
        
        for config_type, result_list in results.items():
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'LOD Performance Analysis: {config_type.replace("_", " ").title()}', fontsize=16, fontweight='bold')
            
            # Extract data
            x_values = []
            x_labels = []
            
            no_lod_init = []
            no_lod_render = []
            no_lod_total = []
            
            lod_init = []
            lod_render = []
            lod_total = []
            
            speedups = []
            point_reductions = []
            final_points = []
            
            for i, result in enumerate(result_list):
                x_values.append(i)
                x_labels.append(result.config.name)
                
                no_lod_init.append(result.no_lod_init_time)
                no_lod_render.append(result.no_lod_render_time)
                no_lod_total.append(result.no_lod_total_time)
                
                lod_init.append(result.lod_init_time)
                lod_render.append(result.lod_render_time)
                lod_total.append(result.lod_total_time)
                
                speedups.append(result.speedup_ratio)
                point_reductions.append(result.point_reduction_pct)
                final_points.append(result.final_points)
            
            # Plot 1: Timing breakdown comparison
            ax1 = axes[0, 0]
            ax1.plot(x_values, no_lod_init, color=colors['no_lod'], linestyle=line_styles['init'], 
                    marker='o', label='No LOD Init', linewidth=2)
            ax1.plot(x_values, no_lod_render, color=colors['no_lod'], linestyle=line_styles['render'], 
                    marker='s', label='No LOD Render', linewidth=2)
            ax1.plot(x_values, lod_init, color=colors['lod'], linestyle=line_styles['init'], 
                    marker='o', label='LOD Init', linewidth=2)
            ax1.plot(x_values, lod_render, color=colors['lod'], linestyle=line_styles['render'], 
                    marker='s', label='LOD Render', linewidth=2)
            
            ax1.set_xlabel('Configuration')
            ax1.set_ylabel('Time (seconds)')
            ax1.set_title('Timing Breakdown: Init vs Render')
            ax1.set_xticks(x_values)
            ax1.set_xticklabels(x_labels, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Total time comparison
            ax2 = axes[0, 1]
            width = 0.35
            x_pos = np.arange(len(x_values))
            
            bars1 = ax2.bar(x_pos - width/2, no_lod_total, width, label='No LOD Total', 
                           color=colors['no_lod'], alpha=0.8)
            bars2 = ax2.bar(x_pos + width/2, lod_total, width, label='LOD Total', 
                           color=colors['lod'], alpha=0.8)
            
            ax2.set_xlabel('Configuration')
            ax2.set_ylabel('Total Time (seconds)')
            ax2.set_title('Total Rendering Time Comparison')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(x_labels, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add speedup annotations
            for i, (bar1, bar2, speedup) in enumerate(zip(bars1, bars2, speedups)):
                height = max(bar1.get_height(), bar2.get_height())
                ax2.annotate(f'{speedup:.1f}x', 
                           xy=(i, height + height*0.05),
                           ha='center', va='bottom', fontweight='bold',
                           color='green' if speedup > 1.1 else 'gray')
            
            # Plot 3: Speedup ratios
            ax3 = axes[1, 0]
            bars = ax3.bar(x_values, speedups, color=[colors['lod'] if s > 1.1 else 'gray' for s in speedups], alpha=0.8)
            ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No Improvement')
            ax3.set_xlabel('Configuration')
            ax3.set_ylabel('Speedup Ratio')
            ax3.set_title('Performance Speedup (Higher is Better)')
            ax3.set_xticks(x_values)
            ax3.set_xticklabels(x_labels, rotation=45, ha='right')
            ax3.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, speedup in zip(bars, speedups):
                height = bar.get_height()
                ax3.annotate(f'{speedup:.2f}x',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
            
            # Plot 4: Point reduction efficiency
            ax4 = axes[1, 1]
            scatter = ax4.scatter(point_reductions, speedups, c=final_points, cmap='viridis', 
                                s=100, alpha=0.7, edgecolors='black', linewidth=1)
            
            ax4.set_xlabel('Point Reduction (%)')
            ax4.set_ylabel('Speedup Ratio')
            ax4.set_title('LOD Efficiency: Point Reduction vs Speedup')
            ax4.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax4)
            cbar.set_label('Final Point Count')
            
            # Annotate points
            for i, (x, y, label) in enumerate(zip(point_reductions, speedups, x_labels)):
                ax4.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.8)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.output_dir / f"lod_performance_{config_type}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  📈 Saved plot: {plot_file}")
    
    def create_summary_report(self, results: Dict[str, List[BenchmarkResult]]):
        """Create a summary report of all benchmark results."""
        print("📋 Creating summary report...")
        
        report_lines = []
        report_lines.append("# LOD Performance Benchmark Report")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        overall_speedups = []
        overall_reductions = []
        
        for config_type, result_list in results.items():
            report_lines.append(f"## {config_type.replace('_', ' ').title()} Analysis")
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
        
        # Overall summary
        report_lines.append("## Overall Performance Summary")
        report_lines.append("")
        report_lines.append(f"- **Overall average speedup**: {np.mean(overall_speedups):.2f}x")
        report_lines.append(f"- **Overall best speedup**: {max(overall_speedups):.2f}x")
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
        
        if np.mean(overall_reductions) > 30:
            report_lines.append("✅ **LOD effectively reduces point cloud complexity**")
        else:
            report_lines.append("⚠️ **LOD point reduction could be more aggressive**")
        
        report_lines.append("")
        
        # Save report
        report_file = self.output_dir / "benchmark_report.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📄 Summary report saved: {report_file}")


def main():
    """Run the complete LOD benchmark suite."""
    print("🚀 Starting comprehensive LOD benchmark with visual reports...")
    
    benchmark = LODBenchmark()
    
    # Run benchmarks
    results = benchmark.run_benchmark_suite()
    
    # Create visualizations
    benchmark.create_performance_plots(results)
    
    # Create summary report
    benchmark.create_summary_report(results)
    
    print(f"\n✅ Benchmark complete! Results saved in: {benchmark.output_dir}")
    print(f"📊 Check the PNG files for visual performance analysis")
    print(f"📄 Check benchmark_report.md for summary")


if __name__ == "__main__":
    main()