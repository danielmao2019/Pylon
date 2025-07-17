# Daily Summary Feature Implementation Plan

## Overview
This document outlines the implementation plan for adding a daily summary feature to the Pylon agent system. The feature will automatically generate comprehensive summaries of experiment runs, resource utilization, and key events that occurred during each day.

## Clarifications

### Resource Bottlenecks
**Definition**: A resource bottleneck occurs when:
- All GPUs on all servers are running at >90% utilization for extended periods (>1 hour)
- CPU load average exceeds the number of cores consistently
- Memory usage prevents new experiments from launching
- Multiple experiments are queued waiting for resources while none are available

### Critical Errors vs Failed Experiments
- **Failed Experiments**: Experiments that stopped running before reaching their target epochs (detected by run_status logic)
- **Critical Errors**: System-level issues that affect the agent infrastructure itself:
  - SSH connection failures to servers
  - SystemMonitor disconnections
  - Agent crashes or exceptions
  - File system issues (disk full, permission errors)

## Current System Understanding

### Agent Module Structure
- **BaseAgent**: Provides core functionality for system monitoring across multiple servers
- **Launcher**: Extends BaseAgent to manage experiment lifecycle (launch, monitor, cleanup)
- **SystemMonitor**: Tracks GPU/CPU usage and running processes
- **RunStatus**: Tracks experiment states (running, finished, failed, stuck, outdated)

### Key Components
1. **Experiment Management**: Automatic launching of failed runs on idle resources
2. **Resource Monitoring**: Real-time GPU/CPU utilization tracking
3. **Status Tracking**: Comprehensive experiment state monitoring
4. **Logging**: TextLogger for structured logging to files and console

## Proposed Daily Summary Feature

### Summary Contents (in order)
1. **Experiment Statistics**
   - Total experiments by status (running, finished, failed, stuck)
   - Experiments completed today (session level)
   - Experiments started today (session level)
   - **Epoch-level reporting**: Newly completed epochs today (per experiment)
   - Failed experiments with error summaries

2. **Key Events**
   - Stuck processes removed (with reasons)
   - Outdated runs cleaned up
   - Critical errors encountered (system-level issues)

3. **Progress Overview**
   - Overall completion percentage across all experiments
   - Experiments nearing completion (>90% complete)
   - Long-running experiments (running for >X days)

4. **Resource Utilization**
   - GPU utilization statistics (average, peak, per-server breakdown)
   - CPU utilization statistics
   - Resource bottlenecks detected
   - Idle resource periods

## Implementation Design

### 1. LogsSnapshot Class
Create a new `utils/automation/logs_snapshot.py`:

```python
class LogsSnapshot:
    def __init__(self, logs_dir: str = "./logs"):
        self.logs_dir = logs_dir
        self.snapshot_dir = os.path.join(logs_dir, "snapshots")
        
    def create_snapshot(self, timestamp: str) -> Dict[str, Any]:
        # Scan all experiment directories and collect state
        snapshot = {
            'timestamp': timestamp,
            'experiments': {},  # {config_path: experiment_state}
            'system_state': self._get_system_state()
        }
        return snapshot
        
    def _get_experiment_state(self, work_dir: str) -> Dict[str, Any]:
        # Collect experiment progress, completion status, epochs
        return {
            'progress': get_session_progress(work_dir, expected_files),
            'status': self._determine_status(work_dir),
            'completed_epochs': self._get_completed_epochs(work_dir),
            'last_update': self._get_last_update(work_dir),
            'total_epochs': self._get_total_epochs(work_dir)
        }
        
    def save_snapshot(self, snapshot: Dict[str, Any], filename: str):
        # Save snapshot to JSON file with date/time name
        
    def load_snapshot(self, filename: str) -> Dict[str, Any]:
        # Load previous snapshot for comparison
```

### 2. DailySummaryGenerator Class
Create summary by comparing snapshots:

```python
class DailySummaryGenerator:
    def __init__(self, config_files: List[str], expected_files: List[str]):
        self.config_files = config_files
        self.expected_files = expected_files
        
    def generate_summary(self, current_snapshot: Dict[str, Any], 
                        previous_snapshot: Dict[str, Any] = None) -> str:
        # Generate summary comparing two snapshots
        
    def _analyze_experiment_changes(self, current, previous) -> Dict[str, Any]:
        # Determine started, completed, failed experiments
        # Calculate newly completed epochs per experiment
        
    def _analyze_key_events(self, current, previous) -> List[str]:
        # Detect stuck removals, cleanups, failures
        
    def _analyze_progress_overview(self, current) -> Dict[str, Any]:
        # Overall completion, near completion, long running
        
    def _analyze_resource_utilization(self, current) -> Dict[str, Any]:
        # Resource usage patterns (if available from system monitor)
```

### 3. Selected Integration Approach: Snapshot-Based Daily Script
- **Crontab-triggered script**: Run daily at 11:55 PM via crontab
- **Snapshot-based approach**: Create daily snapshots of logs directory state
- **Git-like diff generation**: Compare snapshots to generate daily summaries
- **No long-running process**: Script runs, generates summary, sends email, exits
- Sends email notifications upon completion

### 4. Standalone Script Structure
Create `scripts/generate_daily_summary.py`:

```python
#!/usr/bin/env python3
"""
Daily summary generation script for crontab execution.
Usage: python scripts/generate_daily_summary.py --config configs/summary_config.py
"""

def main():
    # 1. Create today's snapshot
    # 2. Load yesterday's snapshot (if exists) 
    # 3. Generate diff-based summary
    # 4. Save summary and email
    # 5. Clean up old snapshots (>30 days)
```

### 5. Storage Format
- **Daily snapshots**: `logs/snapshots/YYYY-MM-DD_HHMMSS.json`
- **Daily summaries**: `logs/summaries/YYYY-MM-DD_summary.md`  
- **Rolling retention**: Keep last 30 days of snapshots and summaries

### 6. Crontab Integration
```bash
# Run daily at 11:55 PM
55 23 * * * cd /path/to/Pylon && python scripts/generate_daily_summary.py --config configs/summary_config.py
```

## Implementation Steps

### Phase 1: Snapshot Infrastructure
1. Create `LogsSnapshot` class to capture logs directory state
2. Implement experiment state collection (progress, epochs, status)
3. Add snapshot saving/loading with timestamped JSON files
4. Test snapshot generation and storage

### Phase 2: Diff-Based Summary Generation  
1. Create `DailySummaryGenerator` class for snapshot comparison
2. Implement experiment change analysis (started, completed, failed)
3. Add epoch-level diff detection for newly completed epochs
4. Generate formatted summary in specified order

### Phase 3: Standalone Script and Email
1. Create `scripts/generate_daily_summary.py` for crontab execution
2. Add email functionality using Python's smtplib
3. Implement cleanup of old snapshots (>30 days retention)
4. Test end-to-end: snapshot → diff → summary → email

### Phase 4: Configuration and Testing
1. Create configuration system for script parameters
2. Add comprehensive error handling and logging
3. Test with existing logs directory structure
4. Document crontab setup and usage

## Configuration Example

```python
summary_config = {
    'enabled': True,
    'summary_time': '23:55',  # When to generate daily summary
    'metrics_interval': 300,   # Collect metrics every 5 minutes
    'summary_retention_days': 30,
    'email': {
        'enabled': True,
        'recipient': 'daniel.mao@uwaterloo.ca'
    },
    'summary_sections': [
        'experiment_statistics',  # Including epoch-level reporting
        'key_events',
        'progress_overview', 
        'resource_utilization'   # Including bottleneck detection
    ]
}
```

## Example Summary Output

```markdown
# Daily Summary - 2024-01-15

## Experiment Statistics
- **Total Active Experiments**: 48
  - Running: 12
  - Finished Today: 15
  - Started Today: 8
  - Failed: 3
  - Stuck (Removed): 2

- **Epoch-Level Progress Today**:
  - configs/exp/baseline.py: Completed epochs 17, 18, 19
  - configs/exp/ablation_1.py: Completed epochs 89, 90, 91, 92
  - configs/exp/model_v2.py: Completed epochs 45, 46
  - configs/exp/transformer.py: Completed epochs 12, 13, 14, 15, 16

## Key Events
- 14:15 - Removed stuck process: configs/exp/model_v2.py (no progress for 3 hours)
- 16:30 - Cleaned 5 outdated runs (>120 days old)
- 18:45 - Failed experiment: configs/exp/model_v3.py (CUDA out of memory)

## Progress Overview
- **Overall Completion**: 73.5% (156/212 experiments)
- **Near Completion** (>90%):
  - configs/exp/baseline.py: 95% (19/20 epochs)
  - configs/exp/ablation_1.py: 92% (92/100 epochs)
- **Long Running** (>7 days):
  - configs/exp/large_model.py: Running for 10 days (45% complete)

## Resource Utilization
- **GPU Usage**:
  - Average: 78.5%
  - Peak: 95.2% (14:32)
  - Idle GPUs: 4/20 (20%)
- **CPU Usage**:
  - Average: 65.3%
  - Peak: 88.1% (15:45)
- **Bottlenecks**: All GPUs >90% utilization from 14:00-16:30 (2.5 hours)
```

## Benefits

1. **Visibility**: Daily insights into experiment progress and system health
2. **Accountability**: Track resource utilization and experiment efficiency
3. **Debugging**: Historical record of issues and events
4. **Planning**: Better resource allocation based on usage patterns
5. **Automation**: Reduce manual monitoring overhead

## Implementation Ready

Based on your requirements, the plan is finalized:

1. **✅ Standalone SummaryAgent** (Option A)
2. **✅ Daily summary at 11:55 PM** with email to daniel.mao@uwaterloo.ca
3. **✅ Content order**: Experiment Stats → Key Events → Progress → Resources  
4. **✅ Epoch-level reporting** for newly completed epochs
5. **✅ No real-time notifications** - daily summary only
6. **✅ Clear definitions** for resource bottlenecks and critical errors

## Advantages of Snapshot-Based Approach

1. **No Long-Running Process**: Script runs, completes task, exits - no hanging processes
2. **Git-Like Version Control**: Compare snapshots to see exactly what changed  
3. **Robust to Interruptions**: Each snapshot is independent, no state loss
4. **Easy Debugging**: Can manually inspect snapshots to understand changes
5. **Scalable**: Snapshot size grows with experiments, not time running
6. **Crontab Integration**: Simple, reliable scheduling mechanism

## Next Steps

I'll now implement:
1. `LogsSnapshot` class for state capture and diff generation
2. `DailySummaryGenerator` for snapshot comparison and summary formatting
3. Standalone script for crontab execution with email functionality
4. Configuration system and error handling
