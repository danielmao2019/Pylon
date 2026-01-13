#!/usr/bin/env python
"""Main entry point for debouncing benchmark suite.

This script provides a command-line interface for running performance benchmarks
comparing viewer callbacks with and without debouncing.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .benchmark_runner import BenchmarkRunner


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark debouncing performance in Pylon data viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available scenarios:
  navigation    - Navigation slider dragging (0 -> 50, 20ms intervals)
  3d_settings   - 3D settings adjustment (point size, opacity)
  mixed         - Mixed interactions (navigation + 3D + transforms)
  stress        - Stress test (100+ events in 1 second)
  buttons       - Rapid button clicking for navigation
  camera        - Camera manipulation and synchronization

Examples:
  # Run full benchmark suite
  python -m benchmarks.data.viewer.debounce.main

  # Run specific scenarios
  python -m benchmarks.data.viewer.debounce.main --scenarios navigation stress

  # Smaller dataset for quick testing
  python -m benchmarks.data.viewer.debounce.main --datapoints 20 --points 1000

        """
    )

    parser.add_argument(
        '--scenarios',
        nargs='+',
        choices=['navigation', '3d_settings', 'mixed', 'stress', 'buttons', 'camera'],
        help='Scenarios to benchmark (default: all scenarios)'
    )

    parser.add_argument(
        '--datapoints',
        type=int,
        default=100,
        help='Number of datapoints in synthetic dataset (default: 100)'
    )

    parser.add_argument(
        '--points',
        type=int,
        default=5000,
        help='Number of points per point cloud (default: 5000)'
    )


    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test mode (20 datapoints, 1000 points)'
    )



    return parser.parse_args()


def validate_args(args):
    """Validate and adjust command line arguments."""
    if args.quick:
        args.datapoints = 20
        args.points = 1000
        print("Quick mode enabled: using 20 datapoints with 1000 points each")

    if args.datapoints < 1:
        print("ERROR: datapoints must be at least 1")
        sys.exit(1)

    if args.points < 100:
        print("ERROR: points must be at least 100")
        sys.exit(1)


def print_banner():
    """Print benchmark suite banner."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      PYLON DATA VIEWER DEBOUNCING BENCHMARK                  ║
║                                                                              ║
║  This benchmark compares the performance of viewer callbacks with and        ║
║  without debouncing to measure the impact on user interaction responsiveness ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


def main():
    """Main entry point for the benchmark suite."""
    args = parse_args()
    validate_args(args)

    print_banner()

    # Create benchmark runner
    runner = BenchmarkRunner()

    # Run benchmark suite
    results = runner.run_full_benchmark(
        scenarios=args.scenarios,
        num_datapoints=args.datapoints,
        num_points=args.points
    )

    # Save results
    output_file = runner.save_results(results)

    # Generate visualizations
    viz_dir = runner.generate_visualizations(results)
    print(f"📈 Visualizations saved to: {viz_dir}")

    # Generate markdown report
    report_path = runner.generate_report(results)
    print(f"📄 Report generated: {report_path}")


if __name__ == "__main__":
    main()