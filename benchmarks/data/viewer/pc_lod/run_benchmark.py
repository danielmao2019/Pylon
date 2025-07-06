#!/usr/bin/env python3
"""Main entry point for the comprehensive LOD benchmark system."""

import argparse

from .orchestrators import SyntheticBenchmarkOrchestrator, RealDataBenchmarkOrchestrator
from .report_generator import ComprehensiveBenchmarkReportGenerator


def main():
    """Run the comprehensive LOD benchmark suite."""
    parser = argparse.ArgumentParser(description="Comprehensive LOD Benchmark Suite")
    parser.add_argument("mode", choices=["synthetic", "real"], 
                       help="Benchmark mode: 'synthetic' (all synthetic tests) or 'real' (all real dataset tests)")
    parser.add_argument("--num-configs", type=int, default=None,
                       help="Number of configurations to test (for quick synthetic testing)")
    
    args = parser.parse_args()
    
    output_dir = "benchmarks/data/viewer/pc_lod/results"
    
    if args.mode == "synthetic":
        print("🚀 Running comprehensive synthetic LOD benchmarks...")
        
        # Initialize components
        synthetic_orchestrator = SyntheticBenchmarkOrchestrator(output_dir)
        report_generator = ComprehensiveBenchmarkReportGenerator(output_dir)
        
        # Run comprehensive synthetic benchmarks across all datasets and camera distances
        print("\n📊 Running comprehensive synthetic benchmark...")
        comprehensive_results = synthetic_orchestrator.run_comprehensive_benchmark(args.num_configs)
        
        # Save all results to JSON
        print("\n💾 Saving comprehensive results...")
        report_generator.save_all_results_json(synthetic_results=comprehensive_results)
        
        # Generate comprehensive plots
        print("\n📊 Generating comprehensive benchmark reports...")
        # First set: Speed vs distance for each dataset
        report_generator.create_speed_vs_distance_by_dataset_plots(comprehensive_results, "synthetic")
        # Second set: Speed vs dataset for each distance
        report_generator.create_speed_vs_dataset_by_distance_plots(comprehensive_results, "synthetic")
        
        # Generate comprehensive summary report
        report_generator.create_comprehensive_summary_report(synthetic_results=comprehensive_results)
        
        print(f"\n✅ Comprehensive synthetic benchmarks complete! Results in: {output_dir}")
        
    elif args.mode == "real":
        print("🚀 Running comprehensive real dataset LOD benchmarks...")
        
        # Initialize components
        real_orchestrator = RealDataBenchmarkOrchestrator(output_dir)
        report_generator = ComprehensiveBenchmarkReportGenerator(output_dir)
        
        # Run real data benchmarks across all datasets and camera distances
        print("\n🏗️ Running comprehensive real data benchmark...")
        real_results = real_orchestrator.run_real_data_benchmark(
            num_samples=10, num_poses_per_distance=3
        )
        
        # Save all results to JSON
        print("\n💾 Saving comprehensive results...")
        report_generator.save_all_results_json(real_data_results=real_results)
        
        # Generate comprehensive plots
        print("\n📊 Generating comprehensive benchmark reports...")
        # First set: Speed vs distance for each dataset
        report_generator.create_speed_vs_distance_by_dataset_plots(real_results, "real")
        # Second set: Speed vs dataset for each distance
        report_generator.create_speed_vs_dataset_by_distance_plots(real_results, "real")
        
        # Generate comprehensive summary report
        report_generator.create_comprehensive_summary_report(real_data_results=real_results)
        
        print(f"\n✅ Comprehensive real data benchmarks complete! Results in: {output_dir}")
    
    print("📊 Check PNG files for comprehensive visualizations")
    print("📄 Check comprehensive_benchmark_report.md for detailed summary")


if __name__ == "__main__":
    main()