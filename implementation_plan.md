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
class ProgressInfo(TypedDict):
    completed_epochs: int
    progress_percentage: float
    early_stopped: bool
    early_stopped_at_epoch: Optional[int]

class RunStatus(NamedTuple):
    config: str
    work_dir: str
    progress: ProgressInfo  # ENHANCED: Rich progress info instead of int
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
        # Create snapshot of logs directory state using enhanced run_status
        snapshot = {
            'timestamp': timestamp,
            'run_statuses': get_all_run_status(  # Use enhanced existing function
                config_files=self.config_files,
                expected_files=self.expected_files,
                epochs=self.epochs,
                sleep_time=self.sleep_time,
                outdated_days=self.outdated_days,
                system_monitor=system_monitor
            )  # Returns Dict[str, RunStatus] with enhanced progress and ProcessInfo
        }
        return snapshot
        
    def save_snapshot(self, snapshot: Dict[str, Any], filename: str):
        os.makedirs(self.snapshot_dir, exist_ok=True)
        filepath = os.path.join(self.snapshot_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(snapshot, f, indent=2, default=str)  # default=str for datetime/etc
            
    def load_snapshot(self, filename: str) -> Optional[Dict[str, Any]]:
        filepath = os.path.join(self.snapshot_dir, filename)
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'r') as f:
            return json.load(f)
```

### 2. Enhanced RunStatus Implementation
Modify existing functions in `utils/automation/run_status.py`:

```python
class ProgressInfo(TypedDict):
    completed_epochs: int
    progress_percentage: float
    early_stopped: bool
    early_stopped_at_epoch: Optional[int]

class RunStatus(NamedTuple):
    config: str
    work_dir: str
    progress: ProgressInfo  # ENHANCED: Rich progress info instead of int
    status: _RunStatus
    process_info: Optional[ProcessInfo]  # NEW: Associated GPU process if running

def get_session_progress(work_dir: str, expected_files: List[str]) -> ProgressInfo:
    """Enhanced to return full progress info instead of just int."""
    # Use existing _compute_and_cache_progress logic but return full dict
    progress_file = os.path.join(work_dir, "progress.json")
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)  # Return full ProgressInfo dict
    
    # Slow path: re-compute and return full progress data
    return _compute_and_cache_progress(work_dir, expected_files)

def get_all_run_status(
    config_files: List[str],
    expected_files: List[str],
    epochs: int,
    sleep_time: int = 86400,
    outdated_days: int = 30,
    system_monitor: SystemMonitor = None,
) -> Dict[str, RunStatus]:  # CHANGED: Return mapping instead of list
    """Enhanced to include ProcessInfo and return mapping."""
    assert isinstance(system_monitor, SystemMonitor)
    
    # Query system monitor ONCE for all GPUs
    all_connected_gpus = system_monitor.connected_gpus
    config_to_process_info = _build_config_to_process_mapping(all_connected_gpus)
    all_running_work_dirs = [get_work_dir(config) for config in config_to_process_info.keys()]
    
    # Get enhanced run status for all configs
    with ThreadPoolExecutor() as executor:
        all_run_status = list(executor.map(
            partial(get_run_status,
                expected_files=expected_files,
                epochs=epochs,
                gpu_running_work_dirs=all_running_work_dirs,
                config_to_process_info=config_to_process_info,
                sleep_time=sleep_time,
                outdated_days=outdated_days,
            ), config_files
        ))
    
    # Convert list to mapping
    return {status.config: status for status in all_run_status}

def get_run_status(
    config: str,
    expected_files: List[str],
    epochs: int,
    gpu_running_work_dirs: List[str],
    config_to_process_info: Dict[str, ProcessInfo],  # NEW parameter
    sleep_time: int = 86400,
    outdated_days: int = 30
) -> RunStatus:
    """Enhanced to include ProcessInfo and rich progress."""
    work_dir = get_work_dir(config)
    
    # Get enhanced progress info (ProgressInfo dict instead of int)
    progress = get_session_progress(work_dir, expected_files)
    
    # Use existing status logic
    log_last_update = get_log_last_update(work_dir)
    epoch_last_update = get_epoch_last_update(work_dir, expected_files)
    is_running_status = log_last_update is not None and (time.time() - log_last_update <= sleep_time)
    
    if is_running_status:
        status = 'running'
    elif progress['completed_epochs'] >= epochs:
        if epoch_last_update is not None and (time.time() - epoch_last_update > outdated_days * 24 * 60 * 60):
            status = 'outdated'
        else:
            status = 'finished'
    elif work_dir in gpu_running_work_dirs:
        status = 'stuck'
    else:
        status = 'failed'
    
    # Get ProcessInfo if this config is running on GPU
    process_info = config_to_process_info.get(config, None)
    
    return RunStatus(
        config=config,
        work_dir=work_dir,
        progress=progress,
        status=status,
        process_info=process_info
    )

def _build_config_to_process_mapping(connected_gpus: List[GPUStatus]) -> Dict[str, ProcessInfo]:
    """Build mapping from config file to ProcessInfo for running experiments."""
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
Create summary by calling LogsSnapshot and adding key events:

```python
class DailySummaryGenerator:
    def __init__(self, 
                 config_files: List[str],
                 expected_files: List[str],
                 epochs: int,
                 sleep_time: int = 86400,
                 outdated_days: int = 30):
        self.logs_snapshot = LogsSnapshot(
            config_files=config_files,
            expected_files=expected_files,
            epochs=epochs,
            sleep_time=sleep_time,
            outdated_days=outdated_days
        )
        self.agent_log_parser = AgentLogParser()
        
    def generate_daily_summary(self, 
                              system_monitor: SystemMonitor,
                              date: str = None) -> str:
        # Create current snapshot
        current_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        current_snapshot = self.logs_snapshot.create_snapshot(current_timestamp, system_monitor)
        
        # Try to load yesterday's snapshot
        yesterday = datetime.now() - timedelta(days=1)
        yesterday_filename = f"{yesterday.strftime('%Y-%m-%d')}_235500.json"
        previous_snapshot = self.logs_snapshot.load_snapshot(yesterday_filename)
        
        # Extract key events separately from agent log
        key_events = self.agent_log_parser.extract_key_events_since_yesterday()
        
        # Generate summary in specified order:
        # 1. Experiment Statistics → 2. Key Events → 3. Progress Overview → 4. Resource Utilization
        sections = [
            f"# Daily Summary - {date or datetime.now().strftime('%Y-%m-%d')}",
            self._format_experiment_statistics(current_snapshot, previous_snapshot),
            self._format_key_events(key_events),
            self._format_progress_overview(current_snapshot),
            self._format_resource_utilization(current_snapshot)
        ]
        
        # Save current snapshot for tomorrow's comparison
        today_filename = f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.json"
        self.logs_snapshot.save_snapshot(current_snapshot, today_filename)
        
        return "\n\n".join(sections)
        
    def _format_experiment_statistics(self, current, previous) -> str:
        current_statuses = current['run_statuses']  # Dict[str, RunStatus]
        previous_statuses = previous['run_statuses'] if previous else {}
        
        # 1. Total experiments by status
        # 2. Experiments completed today (compare status changes)
        # 3. Experiments started today (new ProcessInfo entries)
        # 4. **Epoch-level reporting**: Compare progress.completed_epochs between snapshots
        # 5. Failed experiments with error summaries
        
    def _format_key_events(self, key_events: List[Dict[str, Any]]) -> str:
        # Format events extracted from agent log:
        # - Stuck processes removed (with reasons)
        # - Outdated runs cleaned up
        # - Critical errors encountered (system-level issues)
        
    def _format_progress_overview(self, current) -> str:
        current_statuses = current['run_statuses']  # Dict[str, RunStatus]
        # - Overall completion percentage across all experiments (use progress.progress_percentage)
        # - Experiments nearing completion (>90% complete)
        # - Long-running experiments (using process_info.start_time if available)
        
    def _format_resource_utilization(self, current) -> str:
        # Extract from current_statuses where process_info is not None
        # - Currently running processes per server
        # - Resource distribution across servers
        
    def _calculate_newly_completed_epochs(self, current_statuses, previous_statuses) -> Dict[str, List[int]]:
        # Compare progress.completed_epochs between snapshots to find newly completed epochs
        # Return {config_path: [newly_completed_epoch_numbers]}
        newly_completed = {}
        for config, current_status in current_statuses.items():
            if config in previous_statuses:
                prev_epochs = previous_statuses[config].progress['completed_epochs']
                curr_epochs = current_status.progress['completed_epochs']
                if curr_epochs > prev_epochs:
                    newly_completed[config] = list(range(prev_epochs, curr_epochs))
        return newly_completed
        
    def _detect_long_running_experiments(self, current_statuses) -> List[Dict[str, Any]]:
        # Use process_info.start_time to calculate running duration
        # Return experiments running >7 days with details
        import dateutil.parser
        long_running = []
        for config, status in current_statuses.items():
            if status.process_info and status.status == 'running':
                start_time = dateutil.parser.parse(status.process_info['start_time'])
                runtime = datetime.now() - start_time
                if runtime.days > 7:
                    long_running.append({
                        'config': config,
                        'runtime_days': runtime.days,
                        'progress_percentage': status.progress['progress_percentage'],
                        'server': status.process_info['server'] if 'server' in status.process_info else 'unknown'
                    })
        return long_running
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

### Phase 1: Enhanced RunStatus (Modify Existing Functions)
1. **Enhanced ProgressInfo**: Change `progress: int` to `progress: ProgressInfo` with rich early stopping data
2. **Add ProcessInfo field**: Add `process_info: Optional[ProcessInfo]` to existing RunStatus class
3. **Modify get_session_progress**: Return full ProgressInfo dict instead of just completed_epochs int
4. **Modify get_all_run_status**: Return `Dict[str, RunStatus]` instead of `List[RunStatus]`
5. **Modify get_run_status**: Accept `config_to_process_info` parameter and include ProcessInfo
6. **Efficient GPU querying**: Query system monitor once, build config→ProcessInfo mapping

### Phase 2: Separate Agent Log Parsing
1. **AgentLogParser class**: Create separate parser for `project/run_agent.log`
2. **Key events extraction**: Parse structured log entries (stuck removal, cleanup, errors)
3. **Time-based filtering**: Extract events from yesterday only
4. **Event structuring**: Return standardized event dictionaries

### Phase 3: Snapshot Infrastructure (Logs Only)
1. **LogsSnapshot class**: Simple wrapper around enhanced `get_all_run_status`
2. **Snapshot creation**: Use enhanced run_status that returns mapping with rich progress
3. **Snapshot storage**: JSON files in `agents/snapshots/` with save/load methods
4. **No key events in snapshot**: Keep snapshots for diff-able data only

### Phase 4: Summary Generation (Calls LogsSnapshot)
1. **DailySummaryGenerator**: Takes config parameters, creates LogsSnapshot internally
2. **Snapshot comparison**: Call LogsSnapshot to create current snapshot and load previous
3. **Rich progress analysis**: Use ProgressInfo for detailed epoch-level reporting
4. **Long-running detection**: Use ProcessInfo.start_time for runtime calculation
5. **Key events integration**: Extract separately from agent log and format

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

### Proposed Solution: Enhance Existing Functions
```python
class ProgressInfo(TypedDict):
    completed_epochs: int
    progress_percentage: float
    early_stopped: bool
    early_stopped_at_epoch: Optional[int]

class RunStatus(NamedTuple):
    config: str
    work_dir: str
    progress: ProgressInfo  # ENHANCED: Rich progress instead of int
    status: _RunStatus
    process_info: Optional[ProcessInfo]  # NEW: Associated GPU process if running

# Modify existing get_session_progress to return ProgressInfo dict
def get_session_progress(work_dir: str, expected_files: List[str]) -> ProgressInfo:
    """Enhanced to return full progress info instead of just int."""
    progress_file = os.path.join(work_dir, "progress.json")
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)  # Return full ProgressInfo dict
    
    # Use existing _compute_and_cache_progress which already creates full dict
    return _compute_and_cache_progress(work_dir, expected_files)

# Modify existing get_all_run_status to return mapping and include ProcessInfo
def get_all_run_status(
    config_files: List[str],
    expected_files: List[str],
    epochs: int,
    sleep_time: int = 86400,
    outdated_days: int = 30,
    system_monitor: SystemMonitor = None,
) -> Dict[str, RunStatus]:  # CHANGED: Return mapping instead of list
    """Enhanced to include ProcessInfo and return mapping."""
    # Query all GPUs ONCE (existing logic enhanced)
    all_connected_gpus = system_monitor.connected_gpus
    config_to_process_info = _build_config_to_process_mapping(all_connected_gpus)
    all_running_work_dirs = [get_work_dir(config) for config in config_to_process_info.keys()]
    
    # Use existing ThreadPoolExecutor pattern but with enhanced get_run_status
    with ThreadPoolExecutor() as executor:
        all_run_status = list(executor.map(
            partial(get_run_status,
                expected_files=expected_files,
                epochs=epochs,
                gpu_running_work_dirs=all_running_work_dirs,
                config_to_process_info=config_to_process_info,  # NEW parameter
                sleep_time=sleep_time,
                outdated_days=outdated_days,
            ), config_files
        ))
    
    # Convert list to mapping
    return {status.config: status for status in all_run_status}

# Modify existing get_run_status to accept ProcessInfo mapping and return rich progress
def get_run_status(
    config: str,
    expected_files: List[str],
    epochs: int,
    gpu_running_work_dirs: List[str],
    config_to_process_info: Dict[str, ProcessInfo],  # NEW parameter
    sleep_time: int = 86400,
    outdated_days: int = 30
) -> RunStatus:
    """Enhanced to include ProcessInfo and rich progress."""
    work_dir = get_work_dir(config)
    
    # Use enhanced get_session_progress (returns ProgressInfo dict)
    progress = get_session_progress(work_dir, expected_files)
    
    # Use existing status logic but with enhanced progress
    log_last_update = get_log_last_update(work_dir)
    epoch_last_update = get_epoch_last_update(work_dir, expected_files)
    is_running_status = log_last_update is not None and (time.time() - log_last_update <= sleep_time)
    
    if is_running_status:
        status = 'running'
    elif progress['completed_epochs'] >= epochs:  # Use dict access instead of int
        if epoch_last_update is not None and (time.time() - epoch_last_update > outdated_days * 24 * 60 * 60):
            status = 'outdated'
        else:
            status = 'finished'
    elif work_dir in gpu_running_work_dirs:
        status = 'stuck'
    else:
        status = 'failed'
    
    # Get ProcessInfo from mapping
    process_info = config_to_process_info.get(config, None)
    
    return RunStatus(
        config=config,
        work_dir=work_dir,
        progress=progress,  # Now ProgressInfo dict
        status=status,
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
1. **Enhanced Progress**: Change `progress: int` to `progress: ProgressInfo` with rich early stopping data
2. **ProcessInfo Integration**: Add `process_info: Optional[ProcessInfo]` field to existing RunStatus
3. **Return Mapping**: Change `get_all_run_status` to return `Dict[str, RunStatus]` instead of list
4. **Modify Existing Functions**: Enhance existing functions in `utils/automation/run_status.py`, no new functions
5. **Efficient Querying**: Query all GPUs once, build config→ProcessInfo mapping
6. **Separated Concerns**: 
   - **Snapshots**: For diff-able logs directory state (mapping of enhanced RunStatus)
   - **Key events**: Separate extraction from agent log by time filtering
7. **DailySummaryGenerator calls LogsSnapshot**: Generator creates LogsSnapshot internally and calls it

### Data Flow
1. **System Monitor Query** → Build config→ProcessInfo mapping (once per summary generation)
2. **Enhanced RunStatus** → Use existing functions with ProgressInfo dict + ProcessInfo mapping
3. **LogsSnapshot** → Wrapper around enhanced `get_all_run_status`, returns mapping with rich data
4. **DailySummaryGenerator** → Creates LogsSnapshot, gets current + previous snapshots
5. **Agent Log Parsing** → Extract yesterday's key events separately from `project/run_agent.log`
6. **Summary Generation** → Compare snapshot mappings + format key events
7. **Email Delivery** → Send to daniel.mao@uwaterloo.ca at 11:55 PM via crontab script

## Next Steps

I'll now implement:
1. **Enhanced RunStatus** by modifying existing functions in `utils/automation/run_status.py`:
   - Add `ProgressInfo` TypedDict and change `progress: int` to `progress: ProgressInfo`
   - Add `process_info: Optional[ProcessInfo]` field to RunStatus
   - Modify `get_session_progress` to return full ProgressInfo dict
   - Modify `get_all_run_status` to return `Dict[str, RunStatus]` and include ProcessInfo
   - Modify `get_run_status` to accept ProcessInfo mapping
2. **AgentLogParser** for key events extraction from `project/run_agent.log`
3. **LogsSnapshot** as simple wrapper around enhanced `get_all_run_status`
4. **DailySummaryGenerator** that creates LogsSnapshot internally and compares mappings
5. **Standalone crontab script** with email functionality
