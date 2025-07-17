# Daily Summary Feature Implementation Plan

## Overview
This document outlines the implementation plan for adding a daily summary feature to the Pylon agent system. The feature will automatically generate comprehensive summaries of experiment runs, resource utilization, and key events that occurred during each day.

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

### Summary Contents
1. **Experiment Statistics**
   - Total experiments by status (running, finished, failed, stuck)
   - Experiments completed today
   - Experiments started today
   - Failed experiments with error summaries

2. **Resource Utilization**
   - GPU utilization statistics (average, peak, per-server breakdown)
   - CPU utilization statistics
   - Resource availability trends
   - Idle resource periods

3. **Key Events**
   - Stuck processes removed (with reasons)
   - Outdated runs cleaned up
   - Critical errors encountered
   - Resource bottlenecks

4. **Progress Overview**
   - Overall completion percentage across all experiments
   - Experiments nearing completion (>90% complete)
   - Long-running experiments (running for >X days)
   - Estimated time to completion for active runs

## Implementation Design

### 1. SummaryAgent Class
Create a new `agents/summary_agent.py` that extends BaseAgent:

```python
class SummaryAgent(BaseAgent):
    def __init__(self, ...existing params..., summary_time="23:59"):
        # Initialize daily metrics storage
        self.daily_metrics = DailyMetrics()
        self.summary_time = summary_time
        
    def collect_metrics(self):
        # Collect current state metrics
        
    def generate_daily_summary(self):
        # Generate formatted summary
        
    def save_summary(self, summary):
        # Save to dedicated summary log
```

### 2. DailyMetrics Class
A data structure to accumulate metrics throughout the day:

```python
class DailyMetrics:
    def __init__(self):
        self.experiments_started = []
        self.experiments_completed = []
        self.experiments_failed = []
        self.resource_utilization = []
        self.events = []
        
    def record_experiment_start(self, config, timestamp):
        # Record new experiment launch
        
    def record_resource_snapshot(self, gpu_stats, cpu_stats):
        # Record resource utilization snapshot
```

### 3. Integration Options

#### Option A: Standalone Summary Agent
- Run as a separate process alongside Launcher
- Monitors the same experiment pool independently
- Generates summaries on schedule

#### Option B: Integrated with Launcher
- Add summary functionality directly to Launcher
- Leverage existing monitoring infrastructure
- Generate summaries as part of main loop

#### Option C: Decorator Pattern
- Create a `SummaryLauncher` that wraps standard Launcher
- Intercepts key events for metric collection
- Minimal changes to existing code

### 4. Summary Storage and Distribution

#### Storage Format
- Daily summary files: `logs/summaries/YYYY-MM-DD_summary.md`
- Structured JSON metrics: `logs/summaries/YYYY-MM-DD_metrics.json`
- Rolling retention (keep last 30 days)

#### Distribution Options
- Email notifications (optional)
- Slack/Discord webhooks (optional)
- Web dashboard integration (future)

## Implementation Steps

### Phase 1: Core Implementation
1. Create `DailyMetrics` data structure
2. Implement `SummaryAgent` with basic metric collection
3. Add summary generation logic
4. Implement file-based summary storage

### Phase 2: Integration
1. Choose integration approach (A, B, or C)
2. Modify existing code minimally
3. Add configuration options
4. Test with existing launcher setup

### Phase 3: Enhancement
1. Add visualization capabilities
2. Implement notification system
3. Create summary templates
4. Add historical trend analysis

## Configuration Example

```python
summary_config = {
    'enabled': True,
    'summary_time': '23:59',  # When to generate daily summary
    'metrics_interval': 300,   # Collect metrics every 5 minutes
    'summary_retention_days': 30,
    'notifications': {
        'email': {
            'enabled': False,
            'recipients': []
        },
        'slack': {
            'enabled': False,
            'webhook_url': ''
        }
    },
    'summary_sections': [
        'experiment_statistics',
        'resource_utilization',
        'key_events',
        'progress_overview'
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
  - Failed: 3
  - Stuck (Removed): 2

## Resource Utilization
- **GPU Usage**:
  - Average: 78.5%
  - Peak: 95.2% (14:32)
  - Idle GPUs: 4/20 (20%)

- **CPU Usage**:
  - Average: 65.3%
  - Peak: 88.1% (15:45)

## Key Events
- 14:15 - Removed stuck process: configs/exp/model_v2.py (no progress for 3 hours)
- 16:30 - Cleaned 5 outdated runs (>120 days old)
- 18:45 - Error in configs/exp/model_v3.py: CUDA out of memory

## Progress Overview
- **Overall Completion**: 73.5% (156/212 experiments)
- **Near Completion** (>90%):
  - configs/exp/baseline.py: 95% (19/20 epochs)
  - configs/exp/ablation_1.py: 92% (92/100 epochs)
- **Long Running** (>7 days):
  - configs/exp/large_model.py: Running for 10 days (45% complete)
```

## Benefits

1. **Visibility**: Daily insights into experiment progress and system health
2. **Accountability**: Track resource utilization and experiment efficiency
3. **Debugging**: Historical record of issues and events
4. **Planning**: Better resource allocation based on usage patterns
5. **Automation**: Reduce manual monitoring overhead

## Questions for Discussion

1. Which integration approach (A, B, or C) would you prefer?
2. What additional metrics would be valuable in the daily summary?
3. Should summaries be generated at a fixed time or after a certain period of inactivity?
4. Would you like real-time notifications for critical events?
5. Any specific formatting preferences for the summary output?

## Next Steps

Once we agree on the approach, I'll:
1. Implement the core `SummaryAgent` and `DailyMetrics` classes
2. Add metric collection throughout the day
3. Create the summary generation and formatting logic
4. Integrate with the existing Launcher system
5. Add tests to ensure reliability