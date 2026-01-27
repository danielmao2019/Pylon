"""Generate visualizations for debouncing benchmark results."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional


class BenchmarkVisualizer:
    """Creates visualizations for benchmark results."""

    def __init__(self, results_data: Dict[str, Any]):
        """Initialize with benchmark results data.

        Args:
            results_data: Loaded benchmark results dictionary
        """
        self.results = results_data
        self.scenario_results = results_data.get('scenario_results', {})
        self.summary = results_data.get('summary', {})

        # Set up plotting style
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3

    def create_execution_reduction_chart(self, save_path: Optional[str] = None):
        """Create bar chart showing execution reduction by scenario.

        Args:
            save_path: Path to save the chart (optional)
        """
        scenarios = list(self.scenario_results.keys())
        reductions = []
        executed_without = []
        executed_with = []

        for scenario in scenarios:
            comparison = self.scenario_results[scenario]['comparison']
            exec_data = comparison['execution_reduction']

            reductions.append(exec_data['reduction_percentage'])
            executed_without.append(exec_data['executed_events_without'])
            executed_with.append(exec_data['executed_events_with'])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Left chart: Execution reduction percentages
        colors = plt.cm.RdYlGn([r/100 for r in reductions])
        bars1 = ax1.bar(scenarios, reductions, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Execution Reduction (%)')
        ax1.set_title('Debouncing Execution Reduction by Scenario')
        ax1.set_ylim(0, 100)

        # Add percentage labels on bars
        for bar, reduction in zip(bars1, reductions, strict=True):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{reduction:.1f}%', ha='center', va='bottom', fontweight='bold')

        ax1.tick_params(axis='x', rotation=45)

        # Right chart: Before/after execution counts
        x = np.arange(len(scenarios))
        width = 0.35

        bars2 = ax2.bar(x - width/2, executed_without, width, label='Without Debouncing',
                       color='lightcoral', alpha=0.8, edgecolor='black')
        bars3 = ax2.bar(x + width/2, executed_with, width, label='With Debouncing',
                       color='lightgreen', alpha=0.8, edgecolor='black')

        ax2.set_ylabel('Events Executed')
        ax2.set_title('Events Executed: Before vs After Debouncing')
        ax2.set_xticks(x)
        ax2.set_xticklabels(scenarios, rotation=45)
        ax2.legend()

        # Add count labels on bars
        for bars, counts in [(bars2, executed_without), (bars3, executed_with)]:
            for bar, count in zip(bars, counts, strict=True):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{count}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Execution reduction chart saved to: {save_path}")

        return fig

    def create_time_savings_chart(self, save_path: Optional[str] = None):
        """Create chart showing time savings by scenario.

        Args:
            save_path: Path to save the chart (optional)
        """
        scenarios = list(self.scenario_results.keys())
        time_saved_pct = []
        time_without = []
        time_with = []

        for scenario in scenarios:
            comparison = self.scenario_results[scenario]['comparison']
            time_data = comparison['time_savings']

            time_saved_pct.append(time_data['time_saved_percentage'])
            time_without.append(time_data['total_time_without'])
            time_with.append(time_data['total_time_with'])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Left chart: Time savings percentages
        colors = plt.cm.RdYlGn([max(0, t/100) for t in time_saved_pct])
        bars1 = ax1.bar(scenarios, time_saved_pct, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Time Saved (%)')
        ax1.set_title('Time Savings by Scenario')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # Add percentage labels on bars
        for bar, savings in zip(bars1, time_saved_pct, strict=True):
            height = bar.get_height()
            label_y = height + 1 if height >= 0 else height - 3
            ax1.text(bar.get_x() + bar.get_width()/2., label_y,
                    f'{savings:.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                    fontweight='bold')

        ax1.tick_params(axis='x', rotation=45)

        # Right chart: Before/after execution times
        x = np.arange(len(scenarios))
        width = 0.35

        bars2 = ax2.bar(x - width/2, time_without, width, label='Without Debouncing',
                       color='lightcoral', alpha=0.8, edgecolor='black')
        bars3 = ax2.bar(x + width/2, time_with, width, label='With Debouncing',
                       color='lightgreen', alpha=0.8, edgecolor='black')

        ax2.set_ylabel('Total Time (seconds)')
        ax2.set_title('Execution Time: Before vs After Debouncing')
        ax2.set_xticks(x)
        ax2.set_xticklabels(scenarios, rotation=45)
        ax2.legend()

        # Add time labels on bars
        for bars, times in [(bars2, time_without), (bars3, time_with)]:
            for bar, time_val in zip(bars, times, strict=True):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{time_val:.1f}s', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Time savings chart saved to: {save_path}")

        return fig

    def create_performance_score_chart(self, save_path: Optional[str] = None):
        """Create chart showing overall performance scores by scenario.

        Args:
            save_path: Path to save the chart (optional)
        """
        scenarios = list(self.scenario_results.keys())
        scores = []

        for scenario in scenarios:
            comparison = self.scenario_results[scenario]['comparison']
            scores.append(comparison['performance_score'])

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Color bars based on score (green for positive, red for negative)
        colors = ['lightgreen' if score >= 0 else 'lightcoral' for score in scores]
        bars = ax.bar(scenarios, scores, color=colors, alpha=0.8, edgecolor='black')

        ax.set_ylabel('Performance Score')
        ax.set_title('Overall Performance Score by Scenario\n(Higher is Better)')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        # Add score labels on bars
        for bar, score in zip(bars, scores, strict=True):
            height = bar.get_height()
            label_y = height + 1 if height >= 0 else height - 2
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                   f'{score:.1f}', ha='center', va='bottom' if height >= 0 else 'top',
                   fontweight='bold', fontsize=11)

        ax.tick_params(axis='x', rotation=45)

        # Add average line
        if scores:
            avg_score = sum(scores) / len(scores)
            ax.axhline(y=avg_score, color='blue', linestyle='--', alpha=0.7, linewidth=2)
            ax.text(len(scenarios)-1, avg_score + 2, f'Average: {avg_score:.1f}',
                   ha='right', va='bottom', color='blue', fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance score chart saved to: {save_path}")

        return fig

    def create_callback_breakdown_chart(self, save_path: Optional[str] = None):
        """Create chart showing callback-specific performance breakdown.

        Args:
            save_path: Path to save the chart (optional)
        """
        # Collect callback data across all scenarios
        callback_data = {}

        for scenario_name, scenario_data in self.scenario_results.items():
            comparison = scenario_data['comparison']
            callback_analysis = comparison.get('callback_analysis', {})

            for callback_name, callback_stats in callback_analysis.items():
                if callback_name not in callback_data:
                    callback_data[callback_name] = {
                        'scenarios': [],
                        'executions_without': [],
                        'executions_with': [],
                        'reduction_pct': []
                    }

                callback_data[callback_name]['scenarios'].append(scenario_name)
                callback_data[callback_name]['executions_without'].append(callback_stats['executions_without'])
                callback_data[callback_name]['executions_with'].append(callback_stats['executions_with'])
                callback_data[callback_name]['reduction_pct'].append(callback_stats['execution_reduction_pct'])

        if not callback_data:
            print("No callback-specific data available for visualization")
            return None

        # Create subplots for each callback type
        n_callbacks = len(callback_data)
        fig, axes = plt.subplots(n_callbacks, 1, figsize=(14, 4 * n_callbacks))

        if n_callbacks == 1:
            axes = [axes]

        for i, (callback_name, data) in enumerate(callback_data.items()):
            ax = axes[i]
            scenarios = data['scenarios']

            x = np.arange(len(scenarios))
            width = 0.35

            bars1 = ax.bar(x - width/2, data['executions_without'], width,
                          label='Without Debouncing', color='lightcoral', alpha=0.8, edgecolor='black')
            bars2 = ax.bar(x + width/2, data['executions_with'], width,
                          label='With Debouncing', color='lightgreen', alpha=0.8, edgecolor='black')

            ax.set_ylabel('Executions')
            ax.set_title(f'{callback_name} - Execution Count by Scenario')
            ax.set_xticks(x)
            ax.set_xticklabels(scenarios, rotation=45)
            ax.legend()

            # Add count labels and reduction percentages
            for j, (bar1, bar2, reduction) in enumerate(zip(bars1, bars2, data['reduction_pct'], strict=True)):
                # Count labels
                ax.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height() + 0.5,
                       f'{data["executions_without"][j]}', ha='center', va='bottom', fontsize=9)
                ax.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height() + 0.5,
                       f'{data["executions_with"][j]}', ha='center', va='bottom', fontsize=9)

                # Reduction percentage
                ax.text(j, max(data["executions_without"][j], data["executions_with"][j]) + 2,
                       f'{reduction:.0f}%↓', ha='center', va='bottom',
                       fontweight='bold', color='green' if reduction > 0 else 'red')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Callback breakdown chart saved to: {save_path}")

        return fig

    def create_summary_dashboard(self, save_path: Optional[str] = None):
        """Create a comprehensive dashboard with key metrics.

        Args:
            save_path: Path to save the dashboard (optional)
        """
        if not self.summary:
            print("No summary data available for dashboard")
            return None

        fig = plt.figure(figsize=(16, 12))

        # Create a 3x2 grid for different visualizations
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        scenarios = list(self.scenario_results.keys())

        # 1. Overall metrics (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        metrics = ['Avg Execution\nReduction', 'Avg Time\nSaved', 'Performance\nScore']
        values = [
            self.summary.get('average_execution_reduction_pct', 0),
            self.summary.get('average_time_saved_pct', 0),
            self.summary.get('average_performance_score', 0)
        ]
        colors = ['lightgreen', 'lightblue', 'gold']

        bars = ax1.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_title('Overall Benchmark Results', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Value')

        for bar, value in zip(bars, values, strict=True):
            height = bar.get_height()
            label = f'{value:.1f}%' if bar != bars[2] else f'{value:.1f}'
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    label, ha='center', va='bottom', fontweight='bold', fontsize=11)

        # 2. Execution reduction by scenario (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        reductions = [self.scenario_results[s]['comparison']['execution_reduction']['reduction_percentage']
                     for s in scenarios]
        colors = plt.cm.RdYlGn([r/100 for r in reductions])

        bars = ax2.bar(scenarios, reductions, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_title('Execution Reduction by Scenario', fontweight='bold')
        ax2.set_ylabel('Reduction (%)')
        ax2.tick_params(axis='x', rotation=45)

        for bar, reduction in zip(bars, reductions, strict=True):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{reduction:.0f}%', ha='center', va='bottom', fontsize=9)

        # 3. Time savings by scenario (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        time_savings = [self.scenario_results[s]['comparison']['time_savings']['time_saved_percentage']
                       for s in scenarios]
        colors = plt.cm.RdYlGn([max(0, t/100) for t in time_savings])

        bars = ax3.bar(scenarios, time_savings, color=colors, alpha=0.8, edgecolor='black')
        ax3.set_title('Time Savings by Scenario', fontweight='bold')
        ax3.set_ylabel('Time Saved (%)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        for bar, savings in zip(bars, time_savings, strict=True):
            height = bar.get_height()
            label_y = height + 2 if height >= 0 else height - 4
            ax3.text(bar.get_x() + bar.get_width()/2., label_y,
                    f'{savings:.0f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)

        # 4. Performance scores (middle right)
        ax4 = fig.add_subplot(gs[1, 1])
        scores = [self.scenario_results[s]['comparison']['performance_score'] for s in scenarios]
        colors = ['lightgreen' if score >= 0 else 'lightcoral' for score in scores]

        bars = ax4.bar(scenarios, scores, color=colors, alpha=0.8, edgecolor='black')
        ax4.set_title('Performance Score by Scenario', fontweight='bold')
        ax4.set_ylabel('Score')
        ax4.tick_params(axis='x', rotation=45)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        for bar, score in zip(bars, scores, strict=True):
            height = bar.get_height()
            label_y = height + 1 if height >= 0 else height - 2
            ax4.text(bar.get_x() + bar.get_width()/2., label_y,
                    f'{score:.0f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)

        # 5. Event execution comparison (bottom span)
        ax5 = fig.add_subplot(gs[2, :])

        executed_without = [self.scenario_results[s]['comparison']['execution_reduction']['executed_events_without']
                           for s in scenarios]
        executed_with = [self.scenario_results[s]['comparison']['execution_reduction']['executed_events_with']
                        for s in scenarios]

        x = np.arange(len(scenarios))
        width = 0.35

        bars1 = ax5.bar(x - width/2, executed_without, width, label='Without Debouncing',
                       color='lightcoral', alpha=0.8, edgecolor='black')
        bars2 = ax5.bar(x + width/2, executed_with, width, label='With Debouncing',
                       color='lightgreen', alpha=0.8, edgecolor='black')

        ax5.set_title('Event Execution Comparison', fontweight='bold')
        ax5.set_ylabel('Events Executed')
        ax5.set_xlabel('Scenarios')
        ax5.set_xticks(x)
        ax5.set_xticklabels(scenarios, rotation=45)
        ax5.legend()

        # Add count labels
        for bars, counts in [(bars1, executed_without), (bars2, executed_with)]:
            for bar, count in zip(bars, counts, strict=True):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{count}', ha='center', va='bottom', fontsize=9)

        # Add title and config info
        config = self.results.get('config', {})
        title_text = f"Debouncing Performance Benchmark Dashboard\n"
        title_text += f"Dataset: {config.get('num_datapoints', 'N/A')} datapoints, {config.get('num_points', 'N/A')} points each"
        if config.get('run_date'):
            title_text += f" | Run: {config['run_date']}"

        fig.suptitle(title_text, fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Summary dashboard saved to: {save_path}")

        return fig

    def generate_all_visualizations(self, output_dir: str = "visualizations"):
        """Generate all visualization charts.

        Args:
            output_dir: Directory to save all visualizations
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"Generating visualizations in: {output_path}")

        # Generate each visualization
        self.create_execution_reduction_chart(output_path / "execution_reduction.png")
        plt.close()

        self.create_time_savings_chart(output_path / "time_savings.png")
        plt.close()

        self.create_performance_score_chart(output_path / "performance_scores.png")
        plt.close()

        callback_chart = self.create_callback_breakdown_chart(output_path / "callback_breakdown.png")
        if callback_chart:
            plt.close()

        self.create_summary_dashboard(output_path / "summary_dashboard.png")
        plt.close()

        print(f"✅ All visualizations generated in: {output_path}")


def load_and_visualize(results_file: str, output_dir: str = "visualizations"):
    """Load results file and generate all visualizations.

    Args:
        results_file: Path to benchmark results JSON file
        output_dir: Directory to save visualizations
    """
    try:
        with open(results_file, 'r') as f:
            results_data = json.load(f)

        visualizer = BenchmarkVisualizer(results_data)
        visualizer.generate_all_visualizations(output_dir)

    except FileNotFoundError:
        print(f"Error: Results file not found: {results_file}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in results file: {results_file}")
    except Exception as e:
        print(f"Error generating visualizations: {e}")
