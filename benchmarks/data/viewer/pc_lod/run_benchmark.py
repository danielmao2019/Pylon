#!/usr/bin/env python3
"""Main entry point for the modular LOD benchmark system."""

import argparse

from .orchestrators import SyntheticBenchmarkOrchestrator, RealDataBenchmarkOrchestrator
from .report_generator import BenchmarkReportGenerator


def main():
    """Run the modular LOD benchmark suite."""
    parser = argparse.ArgumentParser(description="LOD Benchmark Suite")
    parser.add_argument("mode", choices=["synthetic", "real"], 
                       help="Benchmark mode: 'synthetic' (all synthetic tests) or 'real' (real dataset tests)")
    parser.add_argument("--data-root", type=str, default=None,
                       help="Path to dataset root directory (required for real mode)")
    parser.add_argument("--num-configs", type=int, default=None,
                       help="Number of configurations to test (for quick testing)")
    
    args = parser.parse_args()
    
    output_dir = "benchmarks/data/viewer/pc_lod/results"
    
    if args.mode == "synthetic":
        print("🚀 Running synthetic LOD benchmarks (quick + comprehensive)...")
        
        # Initialize components
        synthetic_orchestrator = SyntheticBenchmarkOrchestrator(output_dir)
        report_generator = BenchmarkReportGenerator(output_dir)
        
        # Run all synthetic benchmarks
        print("\n⚡ Running quick benchmark...")
        quick_results = synthetic_orchestrator.run_quick_benchmark()
        
        print("\n📊 Running comprehensive benchmark...")
        comprehensive_results = synthetic_orchestrator.run_comprehensive_benchmark(args.num_configs)
        
        # Generate reports
        print("\n📊 Generating synthetic benchmark reports...")
        report_generator.create_synthetic_plots(comprehensive_results)
        report_generator.save_results_json(synthetic_results=comprehensive_results, quick_results=quick_results)
        report_generator.create_summary_report(synthetic_results=comprehensive_results, quick_results=quick_results)
        
        print(f"\n✅ Synthetic benchmarks complete! Results in: {output_dir}")
        
    elif args.mode == "real":
        if args.data_root is None:
            print("❌ --data-root required for real data benchmarks")
            return
            
        print("🚀 Running real dataset LOD benchmarks...")
        
        # Initialize components
        real_orchestrator = RealDataBenchmarkOrchestrator(output_dir)
        report_generator = BenchmarkReportGenerator(output_dir)
        
        # Run real data benchmarks
        print("\n🏗️ Running real data benchmark...")
        real_results = real_orchestrator.run_real_data_benchmark(
            args.data_root, num_samples=10, num_poses_per_distance=3
        )
        
        # Generate reports
        print("\n📊 Generating real data benchmark reports...")
        report_generator.create_real_data_plots(real_results)
        report_generator.save_results_json(real_data_results=real_results)
        report_generator.create_summary_report(real_data_results=real_results)
        
        print(f"\n✅ Real data benchmarks complete! Results in: {output_dir}")
    
    print("📊 Check PNG files for visualizations")
    print("📄 Check modular_benchmark_report.md for summary")


if __name__ == "__main__":
    main()