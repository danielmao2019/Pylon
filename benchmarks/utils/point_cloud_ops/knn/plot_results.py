"""Plotting utilities for KNN benchmark results."""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from datetime import datetime


def plot_results(results: Dict):
    """Create bar plot of benchmark results.

    Args:
        results: Dictionary with benchmark results
    """
    # Setup the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Colors for each method
    colors = {
        "faiss": "#FF6B6B",
        "pytorch3d": "#4ECDC4",
        "torch": "#45B7D1",
        "scipy": "#96CEB4"
    }

    # Bar width and positions
    bar_width = 0.2
    x = np.arange(len(results["size_labels"]))

    # Plot bars for each method
    for i, method in enumerate(results["methods"]):
        if method in results["times"]:
            times = results["times"][method]
            # Replace inf with 0 for plotting (will show as no bar)
            times_plot = [t if t != float('inf') else 0 for t in times]

            # Position for this method's bars
            pos = x + (i - len(results["methods"])/2 + 0.5) * bar_width

            # Plot bars
            bars = ax.bar(
                pos,
                times_plot,
                bar_width,
                label=method,
                color=colors.get(method, "#888888"),
                edgecolor='black',
                linewidth=1
            )

            # Add value labels on bars
            for bar, time in zip(bars, times):
                if time != float('inf') and time > 0:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2,
                        height,
                        f'{time:.3f}',
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        rotation=45
                    )

    # Customize the plot
    ax.set_xlabel('Point Cloud Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')

    # Add date to title
    current_date = datetime.now().strftime("%Y-%m-%d")
    ax.set_title(f'KNN Implementation Performance Comparison - {current_date}\n(Average over 5 shapes × 3 repetitions)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(results["size_labels"])
    ax.legend(loc='upper left', framealpha=0.95, title='Methods', title_fontsize=10)

    # Add grid for better readability
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Keep linear scale for y-axis

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Create results directory
    current_dir = os.path.dirname(__file__)
    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Save the figure
    output_path = os.path.join(results_dir, "benchmark_results.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Show the plot
    plt.show()
