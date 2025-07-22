#!/usr/bin/env python
"""Quick script to display key benchmark results and insights."""

import json
import sys
from pathlib import Path
from typing import Dict, Any


def display_results_summary(results_file: str):
    """Display a formatted summary of benchmark results.
    
    Args:
        results_file: Path to benchmark results JSON file
    """
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"❌ Results file not found: {results_file}")
        return
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in results file: {results_file}")
        return
    
    print("🚀 DEBOUNCING BENCHMARK RESULTS SUMMARY")
    print("=" * 50)
    
    # Configuration info
    config = results.get('config', {})
    print(f"📋 Configuration:")
    print(f"   • Dataset: {config.get('num_datapoints', 'N/A')} datapoints, {config.get('num_points', 'N/A')} points each")
    print(f"   • Scenarios: {', '.join(config.get('scenarios', []))}")
    if config.get('run_date'):
        print(f"   • Run Date: {config['run_date']}")
    print()
    
    # Overall summary
    summary = results.get('summary', {})
    if summary:
        print(f"🎯 OVERALL PERFORMANCE:")
        print(f"   • Average Execution Reduction: {summary.get('average_execution_reduction_pct', 0):.1f}%")
        print(f"   • Average Time Saved: {summary.get('average_time_saved_pct', 0):.1f}%")
        print(f"   • Overall Performance Score: {summary.get('average_performance_score', 0):.1f}")
        print()
        
        # Interpretation
        avg_score = summary.get('average_performance_score', 0)
        if avg_score > 50:
            interpretation = "🎉 EXCELLENT - Debouncing provides significant benefits!"
            recommendation = "Strongly recommend keeping debouncing enabled"
        elif avg_score > 20:
            interpretation = "✅ GOOD - Debouncing provides moderate benefits"
            recommendation = "Recommend keeping debouncing enabled"
        elif avg_score > 0:
            interpretation = "⚠️  MARGINAL - Debouncing provides minimal benefits"
            recommendation = "Consider keeping debouncing for user experience"
        else:
            interpretation = "❌ NEGATIVE - Debouncing may be causing overhead"
            recommendation = "Consider optimizing or disabling debouncing"
        
        print(f"📊 INTERPRETATION: {interpretation}")
        print(f"💡 RECOMMENDATION: {recommendation}")
        print()
    
    # Per-scenario breakdown
    scenario_results = results.get('scenario_results', {})
    if scenario_results:
        print(f"📈 SCENARIO BREAKDOWN:")
        print(f"{'Scenario':<12} {'Exec Reduction':<15} {'Time Saved':<12} {'Score':<8} {'Impact':<15}")
        print("-" * 70)
        
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
                impact = "⚠️  Negative"
            
            print(f"{scenario_name:<12} {exec_reduction:>12.1f}%   {time_saved:>9.1f}%   {score:>6.1f}   {impact:<15}")
        
        print()
    
    # Key insights
    print(f"🔍 KEY INSIGHTS:")
    
    # Find best and worst performing scenarios
    if scenario_results:
        scenarios_by_score = [
            (name, data['comparison']['performance_score'])
            for name, data in scenario_results.items()
        ]
        scenarios_by_score.sort(key=lambda x: x[1], reverse=True)
        
        best_scenario, best_score = scenarios_by_score[0]
        worst_scenario, worst_score = scenarios_by_score[-1]
        
        print(f"   • Best performing scenario: '{best_scenario}' (score: {best_score:.1f})")
        print(f"   • Worst performing scenario: '{worst_scenario}' (score: {worst_score:.1f})")
        
        # Find scenario with highest execution reduction
        scenarios_by_reduction = [
            (name, data['comparison']['execution_reduction']['reduction_percentage'])
            for name, data in scenario_results.items()
        ]
        scenarios_by_reduction.sort(key=lambda x: x[1], reverse=True)
        
        highest_reduction_scenario, highest_reduction = scenarios_by_reduction[0]
        print(f"   • Highest execution reduction: '{highest_reduction_scenario}' ({highest_reduction:.1f}%)")
        
        # Find scenario with highest time savings
        scenarios_by_time = [
            (name, data['comparison']['time_savings']['time_saved_percentage'])
            for name, data in scenario_results.items()
        ]
        scenarios_by_time.sort(key=lambda x: x[1], reverse=True)
        
        highest_time_scenario, highest_time = scenarios_by_time[0]
        print(f"   • Highest time savings: '{highest_time_scenario}' ({highest_time:.1f}%)")
        
        print()
    
    # Check for visualization files
    results_path = Path(results_file)
    viz_dir = results_path.parent / "visualizations"
    
    if viz_dir.exists():
        viz_files = list(viz_dir.glob("*.png"))
        if viz_files:
            print(f"📊 VISUALIZATIONS AVAILABLE:")
            for viz_file in sorted(viz_files):
                print(f"   • {viz_file.name}")
            print(f"   📁 Location: {viz_dir}")
            print()
    
    print("=" * 50)
    print("✨ Analysis complete! Review the visualizations for detailed insights.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        # Look for default results file
        default_paths = [
            "benchmark_results/debounce_benchmark_full.json",
            "./debounce_benchmark_full.json"
        ]
        
        results_file = None
        for path in default_paths:
            if Path(path).exists():
                results_file = path
                break
        
        if not results_file:
            print("Usage: python show_results.py <results_file.json>")
            print("\nOr run from directory containing benchmark_results/debounce_benchmark_full.json")
            sys.exit(1)
    
    display_results_summary(results_file)