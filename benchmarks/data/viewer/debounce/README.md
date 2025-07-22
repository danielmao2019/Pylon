# Debouncing Performance Benchmark

This benchmark suite measures the performance impact of debouncing on Pylon data viewer callbacks by comparing execution with and without debouncing decorators.

## Overview

The benchmark simulates realistic user interaction patterns and measures:
- **Execution reduction**: How many callback executions are prevented by debouncing
- **Time savings**: Reduction in total processing time
- **Resource usage**: CPU and memory consumption differences
- **Per-callback analysis**: Performance breakdown by callback type

## Components

### Core Files

- **`main.py`** - Command-line entry point for running benchmarks
- **`benchmark_runner.py`** - Main benchmark orchestration and result collection
- **`metrics_collector.py`** - Performance metrics collection and analysis
- **`interaction_simulator.py`** - Realistic user interaction simulation
- **`mock_app.py`** - Mock Dash app with viewer callbacks
- **`mock_data.py`** - Synthetic point cloud registration dataset generation
- **`visualizer.py`** - Matplotlib-based visualization generation for results
- **`show_results.py`** - Quick results summary and insights display

### Available Scenarios

- **`navigation`** - Navigation slider dragging (0 → 50, 20ms intervals)
- **`3d_settings`** - 3D settings adjustment (point size, opacity sliders)
- **`mixed`** - Mixed interactions (navigation + 3D settings + transforms)
- **`stress`** - Stress test (100+ events in 1 second)
- **`buttons`** - Rapid button clicking for navigation
- **`camera`** - Camera manipulation and synchronization

## Usage

### Quick Start

```bash
# Run full benchmark suite (all scenarios)
python -m benchmarks.data.viewer.debounce.main

# Run specific scenarios
python -m benchmarks.data.viewer.debounce.main --scenarios navigation stress

# Quick test mode (smaller dataset)
python -m benchmarks.data.viewer.debounce.main --quick
```

### Advanced Usage

```bash
# Custom dataset size
python -m benchmarks.data.viewer.debounce.main --datapoints 50 --points 2000

# Custom output directory
python -m benchmarks.data.viewer.debounce.main --output-dir ./my_results

# Skip visualization generation
python -m benchmarks.data.viewer.debounce.main --no-viz

# Verbose output
python -m benchmarks.data.viewer.debounce.main --verbose
```

### Programmatic Usage

```python
from benchmarks.data.viewer.debounce.benchmark_runner import BenchmarkRunner

# Create runner
runner = BenchmarkRunner(output_dir="./benchmark_results")

# Run single scenario comparison
results = runner.run_comparison('navigation', num_datapoints=100, num_points=5000)

# Run full benchmark suite
full_results = runner.run_full_benchmark()

# Save results and generate visualizations
runner.save_results(full_results)
runner.generate_visualizations(full_results)
```

## Output

### Console Output

The benchmark provides real-time progress information and detailed summaries:

```
============================================================
BENCHMARKING SCENARIO: NAVIGATION
============================================================
Running navigation scenario (WITHOUT debouncing)...
  Events to execute: 51
  Expected duration: 1.0s
  Completed in 2.50s
  Executed: 5/51 events
  Prevention rate: 90.2%

=== DEBOUNCING PERFORMANCE COMPARISON ===

Execution Reduction:
  Without debouncing: 5 events
  With debouncing: 1 events
  Reduction: 4 events (80.0%)

Time Savings:
  Without debouncing: 2.50s
  With debouncing: 2.50s
  Time saved: 0.00s (0.0%)

Overall Performance Score: 26.7
```

### Visualization Output

The benchmark automatically generates comprehensive visualizations:

- **`execution_reduction.png`** - Bar charts showing execution reduction by scenario
- **`time_savings.png`** - Time savings comparison charts  
- **`performance_scores.png`** - Overall performance scores by scenario
- **`callback_breakdown.png`** - Per-callback performance analysis
- **`summary_dashboard.png`** - Comprehensive dashboard with all key metrics

### Quick Results Summary

Use the results summary tool for a formatted overview:

```bash
python benchmarks/data/viewer/debounce/show_results.py benchmark_results/debounce_benchmark_full.json
```

### JSON Results

Results are automatically saved as JSON files containing:

```json
{
  "config": {
    "scenarios": ["navigation", "3d_settings"],
    "num_datapoints": 100,
    "num_points": 5000,
    "run_date": "2025-01-21 10:30:00"
  },
  "scenario_results": {
    "navigation": {
      "without_debounce": { /* execution metrics */ },
      "with_debounce": { /* execution metrics */ },
      "comparison": { /* comparative analysis */ }
    }
  },
  "summary": {
    "average_execution_reduction_pct": 75.2,
    "average_time_saved_pct": 12.4,
    "average_performance_score": 43.8
  }
}
```

## Metrics Explained

### Execution Metrics

- **Total Events**: Number of interaction events in scenario
- **Executed Events**: Number of callbacks that actually executed
- **Prevented Events**: Number of callbacks prevented (by PreventUpdate or debouncing)
- **Prevention Rate**: Percentage of events that were prevented
- **Execution Reduction**: How much debouncing reduced callback executions
- **Performance Score**: Overall score (higher is better) based on execution reduction, time savings, and resource efficiency

### Resource Metrics

- **CPU Usage**: Process CPU utilization during benchmark
- **Memory Usage**: Process memory consumption during benchmark
- **Callback Timings**: Per-callback execution time statistics

## Implementation Details

### Mock Data Generation

The benchmark uses synthetic Point Cloud Registration (PCR) datasets with:
- Configurable number of datapoints and points per cloud
- Realistic point cloud structure (core + shell distribution)
- Ground truth transformations and correspondences
- Multiple transform types (noise, downsampling)

### Realistic Simulation

User interactions are simulated with:
- Realistic timing intervals (10-50ms between events)
- Mixed interaction patterns (slider dragging + button clicks + checkboxes)
- Stress test scenarios for extreme conditions
- Camera manipulation with proper 3D coordinates

### Debouncing Integration

The benchmark tests the actual debouncing implementation:
- Uses the real `@debounce` decorator from `data.viewer.utils.debounce`
- Tests "last call wins" semantics with 1-second delays
- Handles Dash's `PreventUpdate` exception properly
- Maintains thread safety for concurrent callbacks

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're running from the Pylon project root directory
2. **Permission errors**: Check write permissions for output directory
3. **Memory issues**: Reduce `--datapoints` and `--points` for large benchmarks
4. **Timeout errors**: Some scenarios may take longer on slower systems

### Performance Considerations

- Default settings (100 datapoints, 5000 points) provide realistic results but may take several minutes
- Use `--quick` mode for faster testing during development
- Larger point clouds increase processing time but provide more realistic measurements
- System load during benchmarking can affect results - run on idle systems for best accuracy

## Contributing

To add new scenarios:

1. Add scenario logic to `interaction_simulator.py`
2. Register the scenario name in `get_scenario_events()`
3. Update the scenario list in `main.py` argument parser
4. Update this README with scenario description

To extend metrics collection:

1. Add new metrics to `ExecutionMetrics` dataclass
2. Update `analyze_execution_results()` in `metrics_collector.py`
3. Add formatting logic to `format_metrics_summary()`