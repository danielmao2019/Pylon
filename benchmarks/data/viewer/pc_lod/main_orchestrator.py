"""Main orchestrator for the modular LOD benchmark system."""

from typing import Optional

from .orchestrators import SyntheticBenchmarkOrchestrator, RealDataBenchmarkOrchestrator
from .report_generator import BenchmarkReportGenerator


class ModularLODBenchmark:
    """Main orchestrator for the modular LOD benchmark system."""
    
    def __init__(self, output_dir: str = "benchmarks/data/viewer/pc_lod/results"):
        self.output_dir = output_dir
        
        # Initialize components
        self.synthetic_orchestrator = SyntheticBenchmarkOrchestrator(output_dir)
        self.real_data_orchestrator = RealDataBenchmarkOrchestrator(output_dir)
        self.report_generator = BenchmarkReportGenerator(output_dir)
    
    def run_benchmark_suite(self, mode: str = "comprehensive", data_root: Optional[str] = None):
        """Run benchmark suite based on specified mode.
        
        Args:
            mode: "quick", "comprehensive", "real_data", or "all"
            data_root: Path to dataset root directory (required for real_data mode)
        """
        print(f"🚀 Starting modular LOD benchmark in '{mode}' mode...")
        
        synthetic_results = None
        real_data_results = None
        quick_results = None
        
        if mode in ["quick", "all"]:
            print("\\n⚡ Running quick benchmark...")
            quick_results = self.synthetic_orchestrator.run_quick_benchmark()
            
        if mode in ["comprehensive", "all"]:
            print("\\n📊 Running comprehensive synthetic benchmark...")
            synthetic_results = self.synthetic_orchestrator.run_comprehensive_benchmark()
            
        if mode in ["real_data", "all"]:
            if data_root is None:
                print("\\n⚠️ data_root required for real_data mode, skipping...")
            else:
                print("\\n🏗️ Running real data benchmark...")
                real_data_results = self.real_data_orchestrator.run_real_data_benchmark(
                    data_root, num_samples=5, num_poses_per_distance=3
                )
        
        # Generate reports and visualizations
        print("\\n📊 Generating reports and visualizations...")
        
        if synthetic_results:
            self.report_generator.create_synthetic_plots(synthetic_results)
        
        if real_data_results:
            self.report_generator.create_real_data_plots(real_data_results)
        
        # Save results
        self.report_generator.save_results_json(synthetic_results, real_data_results, quick_results)
        
        # Create summary report
        self.report_generator.create_summary_report(synthetic_results, real_data_results, quick_results)
        
        print(f"\\n✅ Modular benchmark complete! Results saved in: {self.output_dir}")
        print(f"📊 Check the PNG files for visual performance analysis")
        print(f"📄 Check modular_benchmark_report.md for summary")