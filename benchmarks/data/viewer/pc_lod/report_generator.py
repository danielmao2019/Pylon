"""Report generation and visualization for LOD benchmarks."""

from typing import Dict, List, Optional
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dataclasses import asdict

from .types import BenchmarkStats


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
            
            stats_text = f"Average Speedup: {avg_speedup:.2f}x\\nAverage Point Reduction: {avg_reduction:.1f}%"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.output_dir / f"real_data_{dataset_name}_performance.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  📈 Saved {dataset_name} plot: {plot_file}")
    
    def save_results_json(self, synthetic_results: Optional[Dict[str, List[BenchmarkStats]]] = None,
                         real_data_results: Optional[Dict[str, Dict[str, List[BenchmarkStats]]]] = None,
                         quick_results: Optional[List[BenchmarkStats]] = None):
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
    
    def create_summary_report(self, synthetic_results: Optional[Dict[str, List[BenchmarkStats]]] = None,
                             real_data_results: Optional[Dict[str, Dict[str, List[BenchmarkStats]]]] = None,
                             quick_results: Optional[List[BenchmarkStats]] = None):
        """Create a comprehensive summary report."""
        print("📋 Creating summary report...")
        
        report_lines = []
        report_lines.append("# Modular LOD Performance Benchmark Report")
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
        report_file = self.output_dir / "modular_benchmark_report.md"
        with open(report_file, 'w') as f:
            f.write('\\n'.join(report_lines))
        
        print(f"📄 Summary report saved: {report_file}")