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
    process_info: Optional[ProcessInfo]  # NEW: Associated GPU process if running
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

### 1. LogsSnapshot Class
Create a new `utils/automation/logs_snapshot.py` for logs directory state only:

```python
class LogsSnapshot:
    def __init__(self, 
                 config_files: List[str],
                 expected_files: List[str],
                 epochs: int,
                 sleep_time: int = 86400,
                 outdated_days: int = 30):
        self.config_files = config_files
        self.expected_files = expected_files
        self.epochs = epochs
        self.sleep_time = sleep_time
        self.outdated_days = outdated_days
        self.snapshot_dir = "./agents/snapshots"
        
    def create_snapshot(self, timestamp: str, system_monitor: SystemMonitor) -> Dict[str, Any]:
        # Create snapshot of logs directory state using existing run_status logic
        snapshot = {
            'timestamp': timestamp,
            'run_statuses': self._get_all_enhanced_run_status(system_monitor)
        }
        return snapshot
        
    def _get_all_enhanced_run_status(self, system_monitor: SystemMonitor) -> List[RunStatus]:
        # Use existing get_all_run_status but with enhanced ProcessInfo
        return get_all_run_status_enhanced(
            config_files=self.config_files,
            expected_files=self.expected_files,
            epochs=self.epochs,
            sleep_time=self.sleep_time,
            outdated_days=self.outdated_days,
            system_monitor=system_monitor
        )
```

### 2. Enhanced RunStatus Implementation
Modify `utils/automation/run_status.py` to include ProcessInfo:

```python
class RunStatus(NamedTuple):
    config: str
    work_dir: str
    progress: int
    status: _RunStatus
    process_info: Optional[ProcessInfo]  # NEW: Associated GPU process if running

def get_all_run_status_enhanced(
    config_files: List[str],
    expected_files: List[str],
    epochs: int,
    sleep_time: int = 86400,
    outdated_days: int = 30,
    system_monitor: SystemMonitor = None,
) -> List[RunStatus]:
    # Query system monitor ONCE for all GPUs
    all_connected_gpus = system_monitor.connected_gpus
    config_to_process_info = _build_config_to_process_mapping(all_connected_gpus)
    
    # Get basic run status for all configs
    with ThreadPoolExecutor() as executor:
        all_run_status = list(executor.map(
            partial(get_run_status_with_process_info,
                expected_files=expected_files,
                epochs=epochs,
                config_to_process_info=config_to_process_info,
                sleep_time=sleep_time,
                outdated_days=outdated_days,
            ), config_files
        ))
    
    return all_run_status

def _build_config_to_process_mapping(connected_gpus: List[GPUStatus]) -> Dict[str, ProcessInfo]:
    # Build mapping from config file to ProcessInfo for running experiments
    config_to_process = {}
    for gpu in connected_gpus:
        for process in gpu['processes']:
            if 'python main.py --config-filepath' in process['cmd']:
                config = parse_config(process['cmd'])
                config_to_process[config] = process
    return config_to_process
```

### 3. Agent Log Parser
Create separate `utils/automation/agent_log_parser.py` for key events:

```python
class AgentLogParser:
    def __init__(self, agent_log_path: str = "./project/run_agent.log"):
        self.agent_log_path = agent_log_path
        
    def extract_key_events_since_yesterday(self) -> List[Dict[str, Any]]:
        # Parse agent log file for yesterday's events
        yesterday = datetime.now() - timedelta(days=1)
        events = []
        
        with open(self.agent_log_path, 'r') as f:
            for line in f:
                event = self._parse_log_line(line)
                if event and event['timestamp'] >= yesterday:
                    events.append(event)
        
        return events
        
    def _parse_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        # Parse structured log entries:
        # - "The following processes will be killed {config: (server, pid)}"
        # - "The following runs has not been updated in the last X days and will be removed"
        # - "Please fix X.py. error_log='...'"
        # - SSH command launches
```

### 4. DailySummaryGenerator Class
Create summary by comparing snapshots and adding key events:

```python
class DailySummaryGenerator:
    def __init__(self):
        self.agent_log_parser = AgentLogParser()
        
    def generate_summary(self, 
                        current_snapshot: Dict[str, Any], 
                        previous_snapshot: Dict[str, Any] = None,
                        date: str = None) -> str:
        # Extract key events separately from agent log
        key_events = self.agent_log_parser.extract_key_events_since_yesterday()
        
        # Generate summary in specified order:
        # 1. Experiment Statistics → 2. Key Events → 3. Progress Overview → 4. Resource Utilization
        sections = [
            f"# Daily Summary - {date}",
            self._format_experiment_statistics(current_snapshot, previous_snapshot),
            self._format_key_events(key_events),
            self._format_progress_overview(current_snapshot),
            self._format_resource_utilization(current_snapshot)
        ]
        
        return "\n\n".join(sections)
        
    def _format_experiment_statistics(self, current, previous) -> str:
        current_statuses = current['run_statuses']
        previous_statuses = previous['run_statuses'] if previous else []
        
        # 1. Total experiments by status
        # 2. Experiments completed today (compare status changes)
        # 3. Experiments started today (new processes with start times)
        # 4. **Epoch-level reporting**: Compare progress between snapshots
        # 5. Failed experiments with error summaries
        
    def _format_key_events(self, key_events: List[Dict[str, Any]]) -> str:
        # Format events extracted from agent log:
        # - Stuck processes removed (with reasons)
        # - Outdated runs cleaned up
        # - Critical errors encountered (system-level issues)
        
    def _format_progress_overview(self, current) -> str:
        current_statuses = current['run_statuses']
        # - Overall completion percentage across all experiments
        # - Experiments nearing completion (>90% complete)
        # - Long-running experiments (using process_info.start_time if available)
        
    def _format_resource_utilization(self, current) -> str:
        # Extract from current_statuses where process_info is not None
        # - GPU utilization statistics (if available from system monitor)
        # - CPU utilization statistics  
        # - Resource bottlenecks detected
        # - Currently running processes per server
        
    def _calculate_newly_completed_epochs(self, current_statuses, previous_statuses) -> Dict[str, List[int]]:
        # Compare progress between snapshots to find newly completed epochs
        # Return {config_path: [newly_completed_epoch_numbers]}
        
    def _detect_long_running_experiments(self, current_statuses) -> List[Dict[str, Any]]:
        # Use process_info.start_time to calculate running duration
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

### Phase 1: Enhanced RunStatus with ProcessInfo
1. **Modify RunStatus**: Add `process_info: Optional[ProcessInfo]` field to existing class
2. **Efficient GPU querying**: Query system monitor once, build config→ProcessInfo mapping
3. **Enhanced get_all_run_status**: Modify existing function to include ProcessInfo without renaming
4. **Status logic refinement**: Use ProcessInfo for better "running" vs "stuck" detection

### Phase 2: Separate Agent Log Parsing
1. **AgentLogParser class**: Create separate parser for `project/run_agent.log`
2. **Key events extraction**: Parse structured log entries (stuck removal, cleanup, errors)
3. **Time-based filtering**: Extract events from yesterday only
4. **Event structuring**: Return standardized event dictionaries

### Phase 3: Snapshot Infrastructure (Logs Only)
1. **LogsSnapshot class**: Create simple snapshot for logs directory state only
2. **Use existing run_status**: Leverage enhanced `get_all_run_status` function
3. **Snapshot storage**: JSON files in `agents/snapshots/` with timestamped names
4. **No key events in snapshot**: Keep snapshots for diff-able data only

### Phase 4: Diff-Based Summary Generation
1. **DailySummaryGenerator**: Create class for snapshot comparison + separate key events
2. **Experiment statistics**: Compare RunStatus lists between snapshots for changes
3. **Key events formatting**: Format separately extracted agent log events
4. **Progress overview**: Use ProcessInfo.start_time for long-running detection
5. **Resource utilization**: Extract from current running processes

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

## Detailed Analysis: Enhanced RunStatus with ProcessInfo

### Problem Statement
The current `run_status.py` determines status based on log updates and GPU queries, but doesn't provide:
1. **Process details** for running experiments (launch time, server, etc.)
2. **Efficient querying** (queries GPU for each config separately)
3. **Long-running detection** (no access to process start times)

### Proposed Solution: Add ProcessInfo Field
```python
class RunStatus(NamedTuple):
    config: str
    work_dir: str
    progress: int
    status: _RunStatus
    process_info: Optional[ProcessInfo]  # NEW: Associated GPU process if running

def get_all_run_status_enhanced(
    config_files: List[str],
    expected_files: List[str],
    epochs: int,
    sleep_time: int = 86400,
    outdated_days: int = 30,
    system_monitor: SystemMonitor = None,
) -> List[RunStatus]:
    # Query all GPUs ONCE
    all_connected_gpus = system_monitor.connected_gpus
    config_to_process_info = _build_config_to_process_mapping(all_connected_gpus)
    all_running_work_dirs = [get_work_dir(config) for config in config_to_process_info.keys()]
    
    # Use existing logic but with ProcessInfo enhancement
    with ThreadPoolExecutor() as executor:
        all_run_status = list(executor.map(
            partial(get_run_status_with_process_info,
                expected_files=expected_files,
                epochs=epochs,
                gpu_running_work_dirs=all_running_work_dirs,
                config_to_process_info=config_to_process_info,
                sleep_time=sleep_time,
                outdated_days=outdated_days,
            ), config_files
        ))
    
    return all_run_status

def get_run_status_with_process_info(
    config: str,
    expected_files: List[str],
    epochs: int,
    gpu_running_work_dirs: List[str],
    config_to_process_info: Dict[str, ProcessInfo],
    sleep_time: int = 86400,
    outdated_days: int = 30
) -> RunStatus:
    # Use existing get_run_status logic
    basic_status = get_run_status(config, expected_files, epochs, 
                                 gpu_running_work_dirs, sleep_time, outdated_days)
    
    # Add ProcessInfo if this config is running on GPU
    process_info = config_to_process_info.get(config, None)
    
    return RunStatus(
        config=basic_status.config,
        work_dir=basic_status.work_dir,
        progress=basic_status.progress,
        status=basic_status.status,
        process_info=process_info
    )
```

### Benefits of ProcessInfo Enhancement
1. **Long-running detection**: Use `process_info.start_time` to calculate runtime
2. **Efficient querying**: Query all GPUs once, not per-config
3. **Server information**: Know which server each experiment is running on
4. **Process details**: PID, user, full command for debugging
5. **Backward compatibility**: Keep existing RunStatus interface, just add field

## Architecture Summary

### Corrected Design Principles
1. **RunStatus Enhancement**: Add `ProcessInfo` field to existing class, keep same name
2. **Efficient Querying**: Query all GPUs once, build config→ProcessInfo mapping
3. **Separated Concerns**: 
   - **Snapshots**: For diff-able logs directory state only
   - **Key events**: Separate extraction from agent log by time filtering
4. **Existing Logic Reuse**: Use existing `get_run_status()` function, just enhance with ProcessInfo
5. **Single-Pass Efficiency**: Query system monitor once, process all configs with that data

### Data Flow
1. **System Monitor Query** → Build config→ProcessInfo mapping
2. **Enhanced RunStatus** → Use existing logic + ProcessInfo for all configs
3. **Snapshot Creation** → Store enhanced RunStatus list with timestamp
4. **Agent Log Parsing** → Extract yesterday's key events separately
5. **Summary Generation** → Compare snapshots + format key events
6. **Email Delivery** → Send to daniel.mao@uwaterloo.ca at 11:55 PM

## Next Steps

I'll now implement:
1. **Enhanced RunStatus** with ProcessInfo field and efficient querying
2. **AgentLogParser** for key events extraction from `project/run_agent.log`
3. **LogsSnapshot** for logs directory state snapshots only
4. **DailySummaryGenerator** with snapshot comparison + separate key events
5. **Standalone crontab script** with email functionality
