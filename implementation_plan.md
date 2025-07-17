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

### 1. SummaryAgent Class
Create a new `agents/summary_agent.py` that extends BaseAgent:

```python
class SummaryAgent(BaseAgent):
    def __init__(self, ...existing params..., 
                 summary_time="23:55",
                 email_recipient="daniel.mao@uwaterloo.ca"):
        super().__init__(...)
        # Initialize daily metrics storage
        self.daily_metrics = DailyMetrics()
        self.summary_time = summary_time
        self.email_recipient = email_recipient
        self.last_epoch_counts = {}  # Track epoch counts for delta calculation
        
    def collect_metrics(self):
        # Collect current state metrics and epoch progress
        
    def generate_daily_summary(self):
        # Generate formatted summary in specified order
        
    def save_and_email_summary(self, summary):
        # Save to dedicated summary log and email
```

### 2. DailyMetrics Class
A data structure to accumulate metrics throughout the day:

```python
class DailyMetrics:
    def __init__(self):
        self.experiments_started = []
        self.experiments_completed = []
        self.experiments_failed = []
        self.epochs_completed = {}  # {config: [epoch_numbers]}
        self.resource_utilization = []
        self.events = []
        self.critical_errors = []
        
    def record_experiment_start(self, config, timestamp):
        # Record new experiment launch
        
    def record_experiment_completion(self, config, timestamp):
        # Record experiment completion
        
    def record_epoch_completion(self, config, epoch_num):
        # Record newly completed epoch
        
    def record_resource_snapshot(self, gpu_stats, cpu_stats):
        # Record resource utilization snapshot
        
    def record_event(self, event_type, details):
        # Record key events (stuck removal, cleanup, etc.)
        
    def record_critical_error(self, error_type, details):
        # Record system-level critical errors
```

### 3. Selected Integration Approach: Standalone Summary Agent
- **Chosen Option A**: Run as a separate process alongside Launcher
- Monitors the same experiment pool independently
- Generates summaries on schedule at 11:55 PM daily
- Sends email notifications upon completion
- No real-time notifications - only daily summaries

### 4. Summary Storage and Distribution

#### Storage Format
- Daily summary files: `logs/summaries/YYYY-MM-DD_summary.md`
- Structured JSON metrics: `logs/summaries/YYYY-MM-DD_metrics.json`
- Rolling retention (keep last 30 days)

#### Distribution
- **Email notifications**: Daily at 11:55 PM to daniel.mao@uwaterloo.ca
- **No real-time notifications**: All events summarized daily only

## Implementation Steps

### Phase 1: Core Implementation
1. Create `DailyMetrics` data structure with epoch-level tracking
2. Implement `SummaryAgent` with basic metric collection
3. Add summary generation logic (experiment stats → key events → progress → resources)
4. Implement file-based summary storage

### Phase 2: Email Integration
1. Add email functionality using Python's smtplib
2. Implement daily scheduling at 11:55 PM
3. Test email delivery to daniel.mao@uwaterloo.ca
4. Add error handling for email failures

### Phase 3: Testing and Deployment
1. Test with existing launcher setup (standalone process)
2. Add configuration options
3. Create comprehensive tests
4. Document usage and configuration

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

## Next Steps

I'll now implement:
1. Core `SummaryAgent` and `DailyMetrics` classes
2. Epoch-level progress tracking
3. Email notification system
4. Daily scheduling mechanism
5. Summary generation with your specified order and content
