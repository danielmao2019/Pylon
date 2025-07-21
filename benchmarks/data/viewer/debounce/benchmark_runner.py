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
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize benchmark runner.
        
        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = Path(output_dir or "benchmark_results")
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


if __name__ == "__main__":
    # Run a quick test benchmark
    run_quick_benchmark('navigation')