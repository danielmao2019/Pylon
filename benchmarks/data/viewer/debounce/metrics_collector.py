"""Collect and analyze performance metrics during benchmark execution."""

import time
import threading
import psutil
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ExecutionMetrics:
    """Performance metrics for a single execution."""
    total_events: int = 0
    executed_events: int = 0
    prevented_events: int = 0
    total_time: float = 0.0
    avg_execution_time: float = 0.0
    max_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    cpu_usage_samples: List[float] = field(default_factory=list)
    memory_usage_samples: List[float] = field(default_factory=list)
    prevention_rate: float = 0.0
    throughput_events_per_sec: float = 0.0
    callback_timings: Dict[str, List[float]] = field(default_factory=dict)


class MetricsCollector:
    """Collects system and performance metrics during benchmark execution."""
    
    def __init__(self, sample_interval: float = 0.05):
        """Initialize metrics collector.
        
        Args:
            sample_interval: Interval between system resource samples (seconds)
        """
        self.sample_interval = sample_interval
        self.collecting = False
        self.collection_thread: Optional[threading.Thread] = None
        
        # Resource tracking
        self.cpu_samples: List[float] = []
        self.memory_samples: List[float] = []
        self.start_time: Optional[float] = None
        
        # Process handle for monitoring
        self.process = psutil.Process(os.getpid())
    
    def start_collection(self):
        """Start collecting system metrics in background thread."""
        if self.collecting:
            return
            
        self.collecting = True
        self.cpu_samples.clear()
        self.memory_samples.clear()
        self.start_time = time.time()
        
        self.collection_thread = threading.Thread(target=self._collection_worker, daemon=True)
        self.collection_thread.start()
    
    def stop_collection(self) -> Dict[str, List[float]]:
        """Stop collecting metrics and return collected data.
        
        Returns:
            Dictionary with CPU and memory usage samples
        """
        self.collecting = False
        
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=1.0)
        
        return {
            'cpu_usage': self.cpu_samples.copy(),
            'memory_usage': self.memory_samples.copy()
        }
    
    def _collection_worker(self):
        """Background worker for collecting system metrics."""
        while self.collecting:
            try:
                # CPU usage (percent)
                cpu_percent = self.process.cpu_percent()
                self.cpu_samples.append(cpu_percent)
                
                # Memory usage (MB)
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                self.memory_samples.append(memory_mb)
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            
            time.sleep(self.sample_interval)
    
    def analyze_execution_results(self, execution_results: Dict[str, Any]) -> ExecutionMetrics:
        """Analyze execution results and compute performance metrics.
        
        Args:
            execution_results: Results from InteractionSimulator.execute_scenario
            
        Returns:
            Analyzed execution metrics
        """
        metrics = ExecutionMetrics()
        
        # Basic counts
        metrics.total_events = execution_results['total_events']
        metrics.executed_events = execution_results['executed_events'] 
        metrics.prevented_events = execution_results['prevented_events']
        metrics.total_time = execution_results['total_scenario_time']
        
        # Prevention rate
        if metrics.total_events > 0:
            metrics.prevention_rate = metrics.prevented_events / metrics.total_events
        
        # Throughput
        if metrics.total_time > 0:
            metrics.throughput_events_per_sec = metrics.total_events / metrics.total_time
        
        # Execution timing analysis
        executed_events = [e for e in execution_results['events'] if e.get('executed', False)]
        
        if executed_events:
            execution_times = [e['execution_time'] for e in executed_events]
            metrics.avg_execution_time = sum(execution_times) / len(execution_times)
            metrics.max_execution_time = max(execution_times)
            metrics.min_execution_time = min(execution_times)
            
            # Group by callback type
            for event in executed_events:
                callback_name = event['callback_name']
                if callback_name not in metrics.callback_timings:
                    metrics.callback_timings[callback_name] = []
                metrics.callback_timings[callback_name].append(event['execution_time'])
        
        # Add system resource data if collected
        system_metrics = self.stop_collection()
        metrics.cpu_usage_samples = system_metrics['cpu_usage']
        metrics.memory_usage_samples = system_metrics['memory_usage']
        
        return metrics
    
    def compute_comparative_metrics(self, with_debounce: ExecutionMetrics, 
                                   without_debounce: ExecutionMetrics) -> Dict[str, Any]:
        """Compare metrics between debounced and non-debounced execution.
        
        Args:
            with_debounce: Metrics from execution with debouncing
            without_debounce: Metrics from execution without debouncing
            
        Returns:
            Comparative analysis dictionary
        """
        comparison = {}
        
        # Event execution comparison
        comparison['execution_reduction'] = {
            'executed_events_with': with_debounce.executed_events,
            'executed_events_without': without_debounce.executed_events,
            'reduction_count': without_debounce.executed_events - with_debounce.executed_events,
            'reduction_percentage': 0.0
        }
        
        if without_debounce.executed_events > 0:
            reduction_pct = (without_debounce.executed_events - with_debounce.executed_events) / without_debounce.executed_events * 100
            comparison['execution_reduction']['reduction_percentage'] = reduction_pct
        
        # Time savings
        comparison['time_savings'] = {
            'total_time_with': with_debounce.total_time,
            'total_time_without': without_debounce.total_time,
            'time_saved': without_debounce.total_time - with_debounce.total_time,
            'time_saved_percentage': 0.0
        }
        
        if without_debounce.total_time > 0:
            time_saved_pct = (without_debounce.total_time - with_debounce.total_time) / without_debounce.total_time * 100
            comparison['time_savings']['time_saved_percentage'] = time_saved_pct
        
        # CPU usage comparison
        if with_debounce.cpu_usage_samples and without_debounce.cpu_usage_samples:
            avg_cpu_with = sum(with_debounce.cpu_usage_samples) / len(with_debounce.cpu_usage_samples)
            avg_cpu_without = sum(without_debounce.cpu_usage_samples) / len(without_debounce.cpu_usage_samples)
            
            comparison['cpu_usage'] = {
                'avg_cpu_with_debounce': avg_cpu_with,
                'avg_cpu_without_debounce': avg_cpu_without,
                'cpu_reduction': avg_cpu_without - avg_cpu_with,
                'cpu_reduction_percentage': (avg_cpu_without - avg_cpu_with) / avg_cpu_without * 100 if avg_cpu_without > 0 else 0
            }
        
        # Memory usage comparison
        if with_debounce.memory_usage_samples and without_debounce.memory_usage_samples:
            avg_mem_with = sum(with_debounce.memory_usage_samples) / len(with_debounce.memory_usage_samples)
            avg_mem_without = sum(without_debounce.memory_usage_samples) / len(without_debounce.memory_usage_samples)
            
            comparison['memory_usage'] = {
                'avg_memory_with_debounce': avg_mem_with,
                'avg_memory_without_debounce': avg_mem_without,
                'memory_reduction': avg_mem_without - avg_mem_with,
                'memory_reduction_percentage': (avg_mem_without - avg_mem_with) / avg_mem_without * 100 if avg_mem_without > 0 else 0
            }
        
        # Callback-specific timing comparison
        callback_comparison = {}
        all_callbacks = set(with_debounce.callback_timings.keys()) | set(without_debounce.callback_timings.keys())
        
        for callback_name in all_callbacks:
            timings_with = with_debounce.callback_timings.get(callback_name, [])
            timings_without = without_debounce.callback_timings.get(callback_name, [])
            
            if timings_with and timings_without:
                avg_with = sum(timings_with) / len(timings_with)
                avg_without = sum(timings_without) / len(timings_without)
                
                callback_comparison[callback_name] = {
                    'executions_with': len(timings_with),
                    'executions_without': len(timings_without), 
                    'avg_time_with': avg_with,
                    'avg_time_without': avg_without,
                    'execution_reduction': len(timings_without) - len(timings_with),
                    'execution_reduction_pct': (len(timings_without) - len(timings_with)) / len(timings_without) * 100 if len(timings_without) > 0 else 0
                }
        
        comparison['callback_analysis'] = callback_comparison
        
        # Overall performance score (higher is better)
        # Based on execution reduction, time savings, and resource efficiency
        exec_score = comparison['execution_reduction']['reduction_percentage']
        time_score = comparison['time_savings']['time_saved_percentage']
        
        cpu_score = 0
        if 'cpu_usage' in comparison:
            cpu_score = comparison['cpu_usage']['cpu_reduction_percentage']
        
        comparison['performance_score'] = (exec_score + time_score + cpu_score) / 3
        
        return comparison


def format_metrics_summary(metrics: ExecutionMetrics, title: str) -> str:
    """Format execution metrics as a readable summary.
    
    Args:
        metrics: Execution metrics to format
        title: Title for the summary section
        
    Returns:
        Formatted summary string
    """
    lines = [f"\n=== {title} ==="]
    
    lines.append(f"Total Events: {metrics.total_events}")
    lines.append(f"Executed Events: {metrics.executed_events}")
    lines.append(f"Prevented Events: {metrics.prevented_events}")
    lines.append(f"Prevention Rate: {metrics.prevention_rate:.1%}")
    lines.append(f"Total Time: {metrics.total_time:.2f}s")
    lines.append(f"Throughput: {metrics.throughput_events_per_sec:.1f} events/sec")
    
    if metrics.executed_events > 0:
        lines.append(f"Avg Execution Time: {metrics.avg_execution_time:.4f}s")
        lines.append(f"Max Execution Time: {metrics.max_execution_time:.4f}s")
        lines.append(f"Min Execution Time: {metrics.min_execution_time:.4f}s")
    
    if metrics.cpu_usage_samples:
        avg_cpu = sum(metrics.cpu_usage_samples) / len(metrics.cpu_usage_samples)
        lines.append(f"Average CPU Usage: {avg_cpu:.1f}%")
    
    if metrics.memory_usage_samples:
        avg_mem = sum(metrics.memory_usage_samples) / len(metrics.memory_usage_samples)
        lines.append(f"Average Memory Usage: {avg_mem:.1f} MB")
    
    # Callback breakdown
    if metrics.callback_timings:
        lines.append("\nCallback Execution Summary:")
        for callback_name, timings in metrics.callback_timings.items():
            avg_time = sum(timings) / len(timings)
            lines.append(f"  {callback_name}: {len(timings)} calls, avg {avg_time:.4f}s")
    
    return "\n".join(lines)


def format_comparison_summary(comparison: Dict[str, Any]) -> str:
    """Format comparative metrics as a readable summary.
    
    Args:
        comparison: Comparison metrics dictionary
        
    Returns:
        Formatted comparison summary string
    """
    lines = ["\n=== DEBOUNCING PERFORMANCE COMPARISON ==="]
    
    # Execution reduction
    exec_data = comparison['execution_reduction']
    lines.append(f"\nExecution Reduction:")
    lines.append(f"  Without debouncing: {exec_data['executed_events_without']} events")
    lines.append(f"  With debouncing: {exec_data['executed_events_with']} events")
    lines.append(f"  Reduction: {exec_data['reduction_count']} events ({exec_data['reduction_percentage']:.1f}%)")
    
    # Time savings
    time_data = comparison['time_savings']
    lines.append(f"\nTime Savings:")
    lines.append(f"  Without debouncing: {time_data['total_time_without']:.2f}s")
    lines.append(f"  With debouncing: {time_data['total_time_with']:.2f}s")
    lines.append(f"  Time saved: {time_data['time_saved']:.2f}s ({time_data['time_saved_percentage']:.1f}%)")
    
    # Resource usage
    if 'cpu_usage' in comparison:
        cpu_data = comparison['cpu_usage']
        lines.append(f"\nCPU Usage:")
        lines.append(f"  Without debouncing: {cpu_data['avg_cpu_without_debounce']:.1f}%")
        lines.append(f"  With debouncing: {cpu_data['avg_cpu_with_debounce']:.1f}%")
        lines.append(f"  CPU reduction: {cpu_data['cpu_reduction']:.1f}% ({cpu_data['cpu_reduction_percentage']:.1f}%)")
    
    if 'memory_usage' in comparison:
        mem_data = comparison['memory_usage']
        lines.append(f"\nMemory Usage:")
        lines.append(f"  Without debouncing: {mem_data['avg_memory_without_debounce']:.1f} MB")
        lines.append(f"  With debouncing: {mem_data['avg_memory_with_debounce']:.1f} MB")
        lines.append(f"  Memory reduction: {mem_data['memory_reduction']:.1f} MB ({mem_data['memory_reduction_percentage']:.1f}%)")
    
    # Performance score
    lines.append(f"\nOverall Performance Score: {comparison['performance_score']:.1f}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test metrics collection
    collector = MetricsCollector()
    
    # Simulate some work
    collector.start_collection()
    time.sleep(2.0)
    metrics_data = collector.stop_collection()
    
    print(f"Collected {len(metrics_data['cpu_usage'])} CPU samples")
    print(f"Collected {len(metrics_data['memory_usage'])} memory samples")