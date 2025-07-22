"""Main benchmark runner for debouncing performance comparison."""

import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict

from .mock_app import create_mock_app
from .interaction_simulator import get_scenario_events, InteractionSimulator
from .metrics_collector import MetricsCollector, ExecutionMetrics, format_metrics_summary, format_comparison_summary


class BenchmarkRunner:
    """Runs debouncing benchmarks and collects performance data."""
    
    def __init__(self):
        """Initialize benchmark runner.
        
        Results are always saved to the 'results' subdirectory.
        """
        # Always use results subdirectory in the benchmark module
        self.output_dir = Path(__file__).parent / "results"
        self.output_dir.mkdir(exist_ok=True)
    
    def run_scenario(self, scenario_name: str, use_debounce: bool, 
                     num_datapoints: int = 100, num_points: int = 5000) -> Dict[str, Any]:
        """Run a single benchmark scenario.
        
        Args:
            scenario_name: Name of scenario to run
            use_debounce: Whether to use debouncing decorators
            num_datapoints: Number of datapoints in synthetic dataset
            num_points: Number of points per point cloud
            
        Returns:
            Dictionary containing execution results and metrics
        """
        # Create mock app with/without debouncing
        app = create_mock_app(
            use_debounce=use_debounce, 
            num_datapoints=num_datapoints,
            num_points=num_points
        )
        
        # Get interaction events for scenario
        events = get_scenario_events(scenario_name, app)
        
        # Set up metrics collection
        metrics_collector = MetricsCollector(sample_interval=0.05)
        simulator = InteractionSimulator(app)
        
        print(f"Running {scenario_name} scenario ({'WITH' if use_debounce else 'WITHOUT'} debouncing)...")
        print(f"  Events to execute: {len(events)}")
        print(f"  Expected duration: {events[-1].timestamp:.1f}s")
        
        # Start metrics collection and execute scenario
        metrics_collector.start_collection()
        execution_results = simulator.execute_scenario(events)
        execution_metrics = metrics_collector.analyze_execution_results(execution_results)
        
        print(f"  Completed in {execution_metrics.total_time:.2f}s")
        print(f"  Executed: {execution_metrics.executed_events}/{execution_metrics.total_events} events")
        print(f"  Prevention rate: {execution_metrics.prevention_rate:.1%}")
        
        return {
            'scenario_name': scenario_name,
            'use_debounce': use_debounce,
            'config': {
                'num_datapoints': num_datapoints,
                'num_points': num_points
            },
            'events': events,
            'execution_results': execution_results,
            'execution_metrics': asdict(execution_metrics)
        }
    
    def run_comparison(self, scenario_name: str, num_datapoints: int = 100, 
                      num_points: int = 5000) -> Dict[str, Any]:
        """Run a scenario with and without debouncing for comparison.
        
        Args:
            scenario_name: Name of scenario to run
            num_datapoints: Number of datapoints in synthetic dataset
            num_points: Number of points per point cloud
            
        Returns:
            Dictionary containing comparison results
        """
        print(f"\n{'='*60}")
        print(f"BENCHMARKING SCENARIO: {scenario_name.upper()}")
        print(f"{'='*60}")
        
        # Run without debouncing
        results_without = self.run_scenario(
            scenario_name, 
            use_debounce=False,
            num_datapoints=num_datapoints,
            num_points=num_points
        )
        
        # Short pause between runs
        time.sleep(0.5)
        
        # Run with debouncing
        results_with = self.run_scenario(
            scenario_name,
            use_debounce=True,
            num_datapoints=num_datapoints, 
            num_points=num_points
        )
        
        # Analyze comparison
        metrics_without = ExecutionMetrics(**results_without['execution_metrics'])
        metrics_with = ExecutionMetrics(**results_with['execution_metrics'])
        
        collector = MetricsCollector()
        comparison = collector.compute_comparative_metrics(metrics_with, metrics_without)
        
        # Print summary
        print(format_metrics_summary(metrics_without, "WITHOUT DEBOUNCING"))
        print(format_metrics_summary(metrics_with, "WITH DEBOUNCING"))
        print(format_comparison_summary(comparison))
        
        return {
            'scenario_name': scenario_name,
            'without_debounce': results_without,
            'with_debounce': results_with,
            'comparison': comparison
        }
    
    def run_full_benchmark(self, scenarios: Optional[List[str]] = None,
                          num_datapoints: int = 100, num_points: int = 5000) -> Dict[str, Any]:
        """Run benchmark on multiple scenarios.
        
        Args:
            scenarios: List of scenario names to run (None for all)
            num_datapoints: Number of datapoints in synthetic dataset
            num_points: Number of points per point cloud
            
        Returns:
            Dictionary containing all benchmark results
        """
        if scenarios is None:
            scenarios = ['navigation', '3d_settings', 'mixed', 'stress', 'buttons', 'camera']
        
        print(f"\n{'='*80}")
        print(f"FULL DEBOUNCING BENCHMARK SUITE")
        print(f"{'='*80}")
        print(f"Scenarios: {', '.join(scenarios)}")
        print(f"Dataset size: {num_datapoints} datapoints, {num_points} points each")
        print(f"Output directory: {self.output_dir}")
        
        full_results = {
            'config': {
                'scenarios': scenarios,
                'num_datapoints': num_datapoints,
                'num_points': num_points,
                'run_date': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'scenario_results': {},
            'summary': {}
        }
        
        scenario_summaries = []
        
        # Run each scenario
        for scenario in scenarios:
            try:
                comparison_results = self.run_comparison(scenario, num_datapoints, num_points)
                full_results['scenario_results'][scenario] = comparison_results
                
                # Collect summary stats
                comparison = comparison_results['comparison']
                scenario_summaries.append({
                    'scenario': scenario,
                    'execution_reduction_pct': comparison['execution_reduction']['reduction_percentage'],
                    'time_saved_pct': comparison['time_savings']['time_saved_percentage'],
                    'performance_score': comparison['performance_score']
                })
                
            except Exception as e:
                print(f"ERROR in scenario {scenario}: {e}")
                full_results['scenario_results'][scenario] = {'error': str(e)}
        
        # Compute overall summary
        if scenario_summaries:
            avg_exec_reduction = sum(s['execution_reduction_pct'] for s in scenario_summaries) / len(scenario_summaries)
            avg_time_saved = sum(s['time_saved_pct'] for s in scenario_summaries) / len(scenario_summaries)
            avg_performance_score = sum(s['performance_score'] for s in scenario_summaries) / len(scenario_summaries)
            
            full_results['summary'] = {
                'scenarios_completed': len(scenario_summaries),
                'average_execution_reduction_pct': avg_exec_reduction,
                'average_time_saved_pct': avg_time_saved,
                'average_performance_score': avg_performance_score,
                'scenario_summaries': scenario_summaries
            }
            
            print(f"\n{'='*80}")
            print(f"BENCHMARK SUMMARY")
            print(f"{'='*80}")
            print(f"Scenarios completed: {len(scenario_summaries)}/{len(scenarios)}")
            print(f"Average execution reduction: {avg_exec_reduction:.1f}%")
            print(f"Average time saved: {avg_time_saved:.1f}%")
            print(f"Average performance score: {avg_performance_score:.1f}")
            print(f"\nPer-scenario breakdown:")
            for summary in scenario_summaries:
                print(f"  {summary['scenario']:12s}: {summary['execution_reduction_pct']:5.1f}% exec reduction, "
                      f"{summary['time_saved_pct']:5.1f}% time saved, score {summary['performance_score']:5.1f}")
        
        return full_results
    
    def generate_report(self, results: Dict[str, Any]) -> Path:
        """Generate a markdown report with benchmark results.
        
        Args:
            results: Benchmark results dictionary
            
        Returns:
            Path to generated report file
        """
        report_path = self.output_dir / "report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Debouncing Benchmark Report\n\n")
            
            # Configuration info
            config = results.get('config', {})
            f.write("## Configuration\n\n")
            f.write(f"- **Dataset**: {config.get('num_datapoints', 'N/A')} datapoints, {config.get('num_points', 'N/A')} points each\n")
            f.write(f"- **Scenarios**: {', '.join(config.get('scenarios', []))}\n")
            if config.get('run_date'):
                f.write(f"- **Run Date**: {config['run_date']}\n")
            f.write("\n")
            
            # Overall summary
            summary = results.get('summary', {})
            if summary:
                f.write("## Overall Performance\n\n")
                f.write(f"- **Average Execution Reduction**: {summary.get('average_execution_reduction_pct', 0):.1f}%\n")
                f.write(f"- **Average Time Saved**: {summary.get('average_time_saved_pct', 0):.1f}%\n")
                f.write(f"- **Overall Performance Score**: {summary.get('average_performance_score', 0):.1f}\n\n")
                
                # Interpretation
                avg_score = summary.get('average_performance_score', 0)
                if avg_score > 50:
                    interpretation = "🎉 **EXCELLENT** - Debouncing provides significant benefits!"
                    recommendation = "Strongly recommend keeping debouncing enabled"
                elif avg_score > 20:
                    interpretation = "✅ **GOOD** - Debouncing provides moderate benefits"
                    recommendation = "Recommend keeping debouncing enabled"
                elif avg_score > 0:
                    interpretation = "⚠️ **MARGINAL** - Debouncing provides minimal benefits"
                    recommendation = "Consider keeping debouncing for user experience"
                else:
                    interpretation = "❌ **NEGATIVE** - Debouncing may be causing overhead"
                    recommendation = "Consider optimizing or disabling debouncing"
                
                f.write("### Interpretation\n\n")
                f.write(f"**{interpretation}**\n\n")
                f.write(f"**Recommendation**: {recommendation}\n\n")
            
            # Per-scenario breakdown
            scenario_results = results.get('scenario_results', {})
            if scenario_results:
                f.write("## Scenario Results\n\n")
                f.write("| Scenario | Exec Reduction | Time Saved | Score | Impact |\n")
                f.write("|----------|----------------|------------|-------|--------|\n")
                
                for scenario_name, scenario_data in scenario_results.items():
                    comparison = scenario_data.get('comparison', {})
                    
                    exec_reduction = comparison.get('execution_reduction', {}).get('reduction_percentage', 0)
                    time_saved = comparison.get('time_savings', {}).get('time_saved_percentage', 0)
                    score = comparison.get('performance_score', 0)
                    
                    # Determine impact level
                    if score > 40:
                        impact = "🔥 High"
                    elif score > 20:
                        impact = "⚡ Medium"
                    elif score > 0:
                        impact = "📈 Low"
                    else:
                        impact = "⚠️ Negative"
                    
                    f.write(f"| {scenario_name} | {exec_reduction:.1f}% | {time_saved:.1f}% | {score:.1f} | {impact} |\n")
                
                f.write("\n")
            
            # Key insights
            f.write("## Key Insights\n\n")
            
            # Find best and worst performing scenarios
            if scenario_results:
                scenarios_by_score = [
                    (name, data['comparison']['performance_score'])
                    for name, data in scenario_results.items()
                ]
                scenarios_by_score.sort(key=lambda x: x[1], reverse=True)
                
                best_scenario, best_score = scenarios_by_score[0]
                worst_scenario, worst_score = scenarios_by_score[-1]
                
                f.write(f"- **Best performing scenario**: `{best_scenario}` (score: {best_score:.1f})\n")
                f.write(f"- **Worst performing scenario**: `{worst_scenario}` (score: {worst_score:.1f})\n")
                
                # Find scenario with highest execution reduction
                scenarios_by_reduction = [
                    (name, data['comparison']['execution_reduction']['reduction_percentage'])
                    for name, data in scenario_results.items()
                ]
                scenarios_by_reduction.sort(key=lambda x: x[1], reverse=True)
                
                highest_reduction_scenario, highest_reduction = scenarios_by_reduction[0]
                f.write(f"- **Highest execution reduction**: `{highest_reduction_scenario}` ({highest_reduction:.1f}%)\n")
                
                # Find scenario with highest time savings
                scenarios_by_time = [
                    (name, data['comparison']['time_savings']['time_saved_percentage'])
                    for name, data in scenario_results.items()
                ]
                scenarios_by_time.sort(key=lambda x: x[1], reverse=True)
                
                highest_time_scenario, highest_time = scenarios_by_time[0]
                f.write(f"- **Highest time savings**: `{highest_time_scenario}` ({highest_time:.1f}%)\n\n")
            
            # Visualizations section
            viz_dir = self.output_dir / "visualizations"
            if viz_dir.exists():
                viz_files = list(viz_dir.glob("*.png"))
                if viz_files:
                    f.write("## Visualizations\n\n")
                    
                    # Map of visualization files to descriptions
                    viz_descriptions = {
                        'execution_reduction.png': 'Execution Reduction by Scenario',
                        'time_savings.png': 'Time Savings Analysis',
                        'performance_scores.png': 'Performance Scores by Scenario',
                        'callback_breakdown.png': 'Per-Callback Performance Analysis',
                        'summary_dashboard.png': 'Comprehensive Performance Dashboard'
                    }
                    
                    for viz_file in sorted(viz_files):
                        filename = viz_file.name
                        description = viz_descriptions.get(filename, filename.replace('_', ' ').title())
                        f.write(f"### {description}\n\n")
                        f.write(f"![{description}](visualizations/{filename})\n\n")
            
            # Performance Score explanation
            f.write("## Performance Score Explanation\n\n")
            f.write("The Performance Score is a composite metric calculated as:\n\n")
            f.write("```\n")
            f.write("performance_score = (execution_reduction_% + time_saved_% + cpu_reduction_%) / 3\n")
            f.write("```\n\n")
            f.write("### Score Interpretation\n\n")
            f.write("| Score Range | Interpretation | Meaning |\n")
            f.write("|-------------|----------------|----------|\n")
            f.write("| 50+ | 🎉 Excellent | Debouncing provides significant benefits |\n")
            f.write("| 20-50 | ✅ Good | Debouncing provides moderate benefits |\n")
            f.write("| 0-20 | ⚠️ Marginal | Debouncing provides minimal benefits |\n")
            f.write("| Below 0 | ❌ Negative | Debouncing may be causing overhead |\n\n")
            
            # Raw data section
            f.write("## Raw Data\n\n")
            f.write(f"Complete benchmark results are available in [`debounce_benchmark_full.json`](debounce_benchmark_full.json)\n\n")
            
            f.write("---\n")
            f.write("*Report generated automatically by the Pylon debouncing benchmark system*\n")
        
        return report_path
    
    def generate_visualizations(self, results: Dict[str, Any]):
        """Generate visualizations for benchmark results.
        
        Args:
            results: Benchmark results dictionary
            
        Returns:
            Path to visualizations directory
        """
        from .visualizer import BenchmarkVisualizer
        
        output_dir = self.output_dir / "visualizations"
        
        visualizer = BenchmarkVisualizer(results)
        visualizer.generate_all_visualizations(str(output_dir))
        
        return output_dir
    
    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> Path:
        """Save benchmark results to JSON file.
        
        Args:
            results: Benchmark results dictionary
            filename: Custom filename (auto-generated if None)
            
        Returns:
            Path to saved results file
        """
        if filename is None:
            if 'scenario_name' in results:
                # Single scenario result
                scenario_name = results['scenario_name']
                filename = f"debounce_benchmark_{scenario_name}.json"
            else:
                # Full benchmark result
                filename = f"debounce_benchmark_full.json"
        
        output_file = self.output_dir / filename
        
        # Convert any non-serializable objects
        def make_serializable(obj):
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            return obj
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=make_serializable)
        
        print(f"\nResults saved to: {output_file}")
        return output_file
    
    def load_results(self, filename: str) -> Dict[str, Any]:
        """Load benchmark results from JSON file.
        
        Args:
            filename: Name of results file to load
            
        Returns:
            Loaded results dictionary
        """
        results_file = self.output_dir / filename
        
        with open(results_file, 'r') as f:
            return json.load(f)


def run_quick_benchmark(scenario: str = 'mixed') -> None:
    """Run a quick single-scenario benchmark for testing.
    
    Args:
        scenario: Name of scenario to run
    """
    runner = BenchmarkRunner()
    results = runner.run_comparison(scenario, num_datapoints=20, num_points=1000)
    runner.save_results(results)
