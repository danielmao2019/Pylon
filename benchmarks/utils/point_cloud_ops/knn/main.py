#!/usr/bin/env python3
"""Main entry point for KNN implementation benchmarking."""

import warnings
from run_benchmark import run_benchmark
from plot_results import plot_results


def main():
    """Main entry point."""
    # Suppress warnings
    warnings.filterwarnings("ignore")

    print("="*60)
    print("KNN Implementation Benchmark")
    print("="*60)

    # Run benchmark
    results = run_benchmark()

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for method in results["methods"]:
        print(f"\n{method}:")
        for size_label, time in zip(results["size_labels"], results["times"][method]):
            if time != float('inf'):
                print(f"  {size_label}: {time:.4f}s")
            else:
                print(f"  {size_label}: FAILED")

    # Create visualization
    print("\nGenerating visualization...")
    plot_results(results)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()