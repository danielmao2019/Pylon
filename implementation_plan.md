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

### Agent Log Analysis Results
**Key Findings from Code Tracing:**
- **Agent log file**: `project/run_agent.log` (not launch.log)
- **Key events structure**: Log contains structured entries for:
  - Stuck process removal: "The following processes will be killed {config: (server, pid)}"
  - Outdated cleanup: "The following runs has not been updated in the last X days and will be removed"
  - Job launching: SSH commands with full experiment details
  - Errors: "Please fix X.py. error_log='...'"

### Long-Running Experiments Analysis
**Two Types of "Running" Status:**
1. **Log-based running**: Recent file updates (within sleep_time) - detected by `get_log_last_update()`
2. **GPU-based running**: Actually running on GPU - detected by `SystemMonitor.get_all_running_commands()`

**Launch Time Sources:**
- **ProcessInfo.start_time**: Available from GPU query (`ps -eo pid=,user=,lstart=,cmd=`)
- **Log file timestamps**: Earliest log file creation time as fallback
- **Work directory creation**: Directory timestamp as last resort

**Proposed RunStatus Enhancement:**
```python
class RunStatus(NamedTuple):
    config: str
    work_dir: str
    progress: int
    status: _RunStatus
    launch_time: Optional[float]  # NEW: Unix timestamp of launch
    is_gpu_running: bool          # NEW: Currently running on GPU
    is_log_running: bool          # NEW: Recent log activity
```

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

### 1. Enhanced LogsSnapshot Class
Create a new `utils/automation/logs_snapshot.py` with single-pass efficiency:

```python
class LogsSnapshot:
    def __init__(self, 
                 logs_dir: str = "./logs",
                 config_files: List[str] = None,
                 expected_files: List[str] = None,
                 agent_log_path: str = "./project/run_agent.log"):
        self.logs_dir = logs_dir
        self.snapshot_dir = "./agents/snapshots"
        self.config_files = config_files
        self.expected_files = expected_files
        self.agent_log_path = agent_log_path
        
    def create_snapshot(self, timestamp: str) -> Dict[str, Any]:
        # SINGLE-PASS collection of all information
        snapshot = {
            'timestamp': timestamp,
            'experiments': {},  # {config_path: enhanced_experiment_state}
            'key_events': self._extract_key_events_since_yesterday(),
            'system_state': self._get_system_state_with_launch_times()
        }
        
        # Single pass through logs directory
        self._scan_logs_directory_once(snapshot)
        return snapshot
        
    def _scan_logs_directory_once(self, snapshot: Dict[str, Any]):
        # Walk logs directory once, collect all experiment states
        for config in self.config_files:
            work_dir = get_work_dir(config)
            if os.path.exists(work_dir):
                snapshot['experiments'][config] = self._get_enhanced_experiment_state(work_dir)
                
    def _get_enhanced_experiment_state(self, work_dir: str) -> Dict[str, Any]:
        # Enhanced state with launch time and detailed epoch info
        return {
            'progress': get_session_progress(work_dir, self.expected_files),
            'status': self._determine_status(work_dir),
            'completed_epochs': self._get_completed_epochs_list(work_dir),
            'last_update': self._get_last_update(work_dir),
            'total_epochs': self._get_total_epochs(work_dir),
            'launch_time': self._get_launch_time(work_dir),
            'is_gpu_running': False,  # Will be updated by system monitor
            'is_log_running': self._is_log_running(work_dir)
        }
        
    def _extract_key_events_since_yesterday(self) -> List[Dict[str, Any]]:
        # Parse agent log file for yesterday's events
        # Return structured list of events with timestamps
        
    def _get_system_state_with_launch_times(self) -> Dict[str, Any]:
        # Query SystemMonitor and augment with process launch times
```

### 2. DailySummaryGenerator Class
Create summary by comparing snapshots with specified content order:

```python
class DailySummaryGenerator:
    def __init__(self, config_files: List[str], expected_files: List[str]):
        self.config_files = config_files
        self.expected_files = expected_files
        
    def generate_summary(self, current_snapshot: Dict[str, Any], 
                        previous_snapshot: Dict[str, Any] = None) -> str:
        # Generate summary in specified order:
        # 1. Experiment Statistics → 2. Key Events → 3. Progress Overview → 4. Resource Utilization
        
        sections = []
        sections.append(self._format_experiment_statistics(current_snapshot, previous_snapshot))
        sections.append(self._format_key_events(current_snapshot))
        sections.append(self._format_progress_overview(current_snapshot))
        sections.append(self._format_resource_utilization(current_snapshot))
        
        return "\n\n".join(sections)
        
    def _format_experiment_statistics(self, current, previous) -> str:
        # 1. Total experiments by status (running, finished, failed, stuck)
        # 2. Experiments completed today (session level)
        # 3. Experiments started today (session level) 
        # 4. **Epoch-level reporting**: Newly completed epochs today (per experiment)
        # 5. Failed experiments with error summaries
        
    def _format_key_events(self, current) -> str:
        # Extract from current_snapshot['key_events'] (parsed from agent log):
        # - Stuck processes removed (with reasons)
        # - Outdated runs cleaned up
        # - Critical errors encountered (system-level issues)
        
    def _format_progress_overview(self, current) -> str:
        # - Overall completion percentage across all experiments
        # - Experiments nearing completion (>90% complete)
        # - Long-running experiments (running for >X days using launch_time)
        
    def _format_resource_utilization(self, current) -> str:
        # - GPU utilization statistics (average, peak, per-server breakdown)
        # - CPU utilization statistics  
        # - Resource bottlenecks detected
        # - Idle resource periods
        
    def _calculate_newly_completed_epochs(self, current, previous) -> Dict[str, List[int]]:
        # Compare completed_epochs between snapshots to find newly completed epochs
        # Return {config_path: [newly_completed_epoch_numbers]}
        
    def _detect_long_running_experiments(self, current) -> List[Dict[str, Any]]:
        # Use launch_time to calculate running duration
        # Return experiments running >7 days with details
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
- **Daily snapshots**: `agents/snapshots/YYYY-MM-DD_HHMMSS.json`
- **Daily summaries**: `agents/summaries/YYYY-MM-DD_summary.md`  
- **Rolling retention**: Keep last 30 days of snapshots and summaries

### 6. Crontab Integration
```bash
# Run daily at 11:55 PM
55 23 * * * cd /path/to/Pylon && python scripts/generate_daily_summary.py --config configs/summary_config.py
```

## Implementation Steps

### Phase 1: Enhanced RunStatus and Agent Log Parsing
1. **Augment RunStatus**: Add `launch_time`, `is_gpu_running`, `is_log_running` fields
2. **Agent log parser**: Create parser for `project/run_agent.log` to extract key events
3. **Launch time detection**: Implement logic to get process start times from SystemMonitor
4. **Single-pass efficiency**: Design logs directory scanning to collect all info in one pass

### Phase 2: Snapshot Infrastructure
1. **Enhanced LogsSnapshot**: Create class with single-pass directory scanning
2. **Experiment state collection**: Implement enhanced state with launch times and epoch lists
3. **Key events integration**: Merge agent log events into snapshot structure
4. **Snapshot storage**: JSON files in `agents/snapshots/` with timestamped names

### Phase 3: Diff-Based Summary Generation
1. **DailySummaryGenerator**: Create class for snapshot comparison with specified content order
2. **Experiment statistics**: Implement session-level and epoch-level change detection
3. **Key events formatting**: Format agent log events for summary output
4. **Progress overview**: Add long-running detection using launch times
5. **Resource utilization**: Format GPU/CPU stats and bottleneck detection

### Phase 4: Standalone Script and Email
1. **Daily script**: Create `scripts/generate_daily_summary.py` for crontab execution
2. **Email integration**: Add Python smtplib functionality for daniel.mao@uwaterloo.ca
3. **Cleanup logic**: Implement 30-day retention for snapshots and summaries
4. **Error handling**: Robust error handling and logging for production use

### Phase 5: Testing and Documentation
1. **End-to-end testing**: Test complete pipeline with real logs directory
2. **Crontab setup**: Document installation and configuration
3. **Performance validation**: Ensure single-pass efficiency works at scale
4. **Email delivery testing**: Verify email formatting and delivery

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

## Detailed Analysis: Two Types of "Running" Status

### Problem Statement
The current `run_status.py` determines if an experiment is "running" based on recent log updates, but this doesn't distinguish between:
1. **Actually running on GPU** (process exists in GPU query)
2. **Recently active** (logs updated within sleep_time)

### Proposed Solution: Enhanced Status Detection
```python
def get_enhanced_run_status(
    config: str,
    expected_files: List[str], 
    epochs: int,
    system_monitor: SystemMonitor,
    sleep_time: int = 86400,
    outdated_days: int = 30
) -> EnhancedRunStatus:
    
    work_dir = get_work_dir(config)
    
    # Get timing information
    log_last_update = get_log_last_update(work_dir)
    epoch_last_update = get_epoch_last_update(work_dir, expected_files)
    
    # Check if running on GPU
    running_commands = system_monitor.get_all_running_commands()
    is_gpu_running = any(config in cmd for cmd in running_commands)
    
    # Check if recently active (log-based)
    is_log_running = (log_last_update is not None and 
                     (time.time() - log_last_update <= sleep_time))
    
    # Get launch time if running on GPU
    launch_time = None
    if is_gpu_running:
        launch_time = _get_process_launch_time(config, system_monitor)
    elif is_log_running:
        # Fallback to earliest log timestamp
        launch_time = _get_earliest_log_timestamp(work_dir)
    
    # Determine final status
    if is_gpu_running and is_log_running:
        status = 'running'
    elif is_gpu_running and not is_log_running:
        status = 'stuck'  # Running on GPU but no recent logs
    elif not is_gpu_running and is_log_running:
        status = 'recently_active'  # Recent logs but not on GPU (may have just finished)
    elif progress >= epochs:
        status = 'finished' if not _is_outdated(epoch_last_update, outdated_days) else 'outdated'
    else:
        status = 'failed'
    
    return EnhancedRunStatus(
        config=config,
        work_dir=work_dir,
        progress=progress,
        status=status,
        launch_time=launch_time,
        is_gpu_running=is_gpu_running,
        is_log_running=is_log_running
    )
```

### Long-Running Experiment Detection
With launch times available, we can detect:
- **Very long experiments**: Running >7 days (may need investigation)
- **Stuck experiments**: On GPU but no log updates (immediate attention)
- **Efficiency analysis**: Time per epoch for performance insights

### Benefits of Enhanced Status
1. **Better debugging**: Distinguish between truly stuck vs recently finished
2. **Resource management**: Identify experiments hogging GPUs without progress  
3. **Performance tracking**: Monitor experiment efficiency over time
4. **Accurate reporting**: More precise status in daily summaries

## Next Steps

I'll now implement:
1. **Enhanced RunStatus** with launch time and dual running status detection
2. **Agent log parser** for key events extraction from `project/run_agent.log`
3. **Single-pass LogsSnapshot** for efficient data collection
4. **DailySummaryGenerator** with epoch-level diff detection and specified content order
5. **Standalone crontab script** with email functionality
