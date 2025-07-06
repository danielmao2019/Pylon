#!/usr/bin/env python3
"""Main entry point for the modular LOD benchmark system."""

import argparse

from .main_orchestrator import ModularLODBenchmark


def main():
    """Run the modular LOD benchmark suite."""
    parser = argparse.ArgumentParser(description="Modular LOD Benchmark Suite")
    parser.add_argument("--mode", choices=["quick", "comprehensive", "real_data", "all"], 
                       default="comprehensive",
                       help="Benchmark mode to run (default: comprehensive)")
    parser.add_argument("--data-root", type=str, default=None,
                       help="Path to dataset root directory (required for real_data mode)")
    
    args = parser.parse_args()
    
    print("🚀 Starting modular LOD benchmark with clean architecture...")
    
    benchmark = ModularLODBenchmark()
    benchmark.run_benchmark_suite(mode=args.mode, data_root=args.data_root)


if __name__ == "__main__":
    main()