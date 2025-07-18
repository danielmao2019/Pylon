from typing import List, Dict, Any, Optional
import os
from datetime import datetime, timedelta
from utils.monitor.system_monitor import SystemMonitor
from agents.logs_snapshot import LogsSnapshot
from agents.agent_log_parser import AgentLogParser


class DailySummaryGenerator:
    """Generates comprehensive daily summaries of experiment progress and system events.
    
    Combines logs directory snapshots with agent log events to create detailed reports
    covering experiment statistics, key events, progress overview, and resource utilization.
    """
    
    def __init__(self, 
                 config_files: List[str],
                 expected_files: List[str],
                 epochs: int,
                 sleep_time: int = 86400,
                 outdated_days: int = 30,
                 agent_log_path: str = "./project/run_agent.log"):
        """Initialize daily summary generator.
        
        Args:
            config_files: List of config file paths to monitor
            expected_files: List of expected file patterns per epoch
            epochs: Total number of epochs expected for experiments
            sleep_time: Time to wait for status updates (seconds)
            outdated_days: Number of days to consider a run outdated
            agent_log_path: Path to agent log file for event extraction
        """
        assert isinstance(config_files, list), f"config_files must be list, got {type(config_files)}"
        assert isinstance(expected_files, list), f"expected_files must be list, got {type(expected_files)}"
        assert isinstance(epochs, int), f"epochs must be int, got {type(epochs)}"
        assert isinstance(sleep_time, int), f"sleep_time must be int, got {type(sleep_time)}"
        assert isinstance(outdated_days, int), f"outdated_days must be int, got {type(outdated_days)}"
        assert isinstance(agent_log_path, str), f"agent_log_path must be str, got {type(agent_log_path)}"
        
        self.logs_snapshot = LogsSnapshot(
            config_files=config_files,
            expected_files=expected_files,
            epochs=epochs,
            sleep_time=sleep_time,
            outdated_days=outdated_days
        )
        self.agent_log_parser = AgentLogParser(agent_log_path=agent_log_path)
        self.summary_dir = "./agents/summaries"
        
    def generate_daily_summary(self, 
                              system_monitor: SystemMonitor,
                              date: Optional[str] = None) -> str:
        """Generate comprehensive daily summary.
        
        Args:
            system_monitor: SystemMonitor instance for current state queries
            date: Date string (YYYY-MM-DD) for the summary, defaults to today
            
        Returns:
            Formatted markdown summary string
        """
        assert isinstance(system_monitor, SystemMonitor), f"system_monitor must be SystemMonitor, got {type(system_monitor)}"
        
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # Create current snapshot
        current_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        current_snapshot = self.logs_snapshot.create_snapshot(current_timestamp, system_monitor)
        
        # Try to load yesterday's snapshot for comparison
        yesterday = datetime.now() - timedelta(days=1)
        yesterday_filename = f"{yesterday.strftime('%Y-%m-%d')}_235500.json"
        previous_snapshot = self.logs_snapshot.load_snapshot(yesterday_filename)
        
        # Extract key events from agent log
        key_events = self.agent_log_parser.extract_key_events_since_yesterday()
        
        # Generate summary sections in specified order
        sections = [
            f"# Daily Summary - {date}",
            "",
            self._format_experiment_statistics(current_snapshot, previous_snapshot),
            "",
            self._format_key_events(key_events),
            "",
            self._format_progress_overview(current_snapshot),
            "",
            self._format_resource_utilization(current_snapshot)
        ]
        
        # Save current snapshot for tomorrow's comparison
        today_filename = f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.json"
        self.logs_snapshot.save_snapshot(current_snapshot, today_filename)
        
        return "\\n".join(sections)
    
    def _format_experiment_statistics(self, current: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> str:
        """Format experiment statistics section.
        
        Args:
            current: Current snapshot data
            previous: Previous snapshot data (may be None)
            
        Returns:
            Formatted experiment statistics section
        """
        current_statuses = current['run_statuses']  # Dict[str, RunStatus]
        previous_statuses = previous['run_statuses'] if previous else {}
        
        # Count experiments by status
        status_counts = {}
        for run_status in current_statuses.values():
            status = run_status['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Calculate experiments completed and started today
        completed_today = self._find_experiments_completed_today(current_statuses, previous_statuses)
        started_today = self._find_experiments_started_today(current_statuses, previous_statuses)
        
        # Calculate newly completed epochs
        newly_completed_epochs = self._calculate_newly_completed_epochs(current_statuses, previous_statuses)
        
        # Find failed experiments
        failed_experiments = self._find_failed_experiments(current_statuses, previous_statuses)
        
        lines = [
            "## Experiment Statistics",
            f"- **Total Active Experiments**: {len(current_statuses)}",
            f"  - Running: {status_counts.get('running', 0)}",
            f"  - Finished: {status_counts.get('finished', 0)}",
            f"  - Failed: {status_counts.get('failed', 0)}",
            f"  - Stuck: {status_counts.get('stuck', 0)}",
            f"  - Outdated: {status_counts.get('outdated', 0)}",
            "",
            f"- **Experiments Finished Today**: {len(completed_today)}",
            f"- **Experiments Started Today**: {len(started_today)}",
        ]
        
        # Add epoch-level progress if there are newly completed epochs
        if newly_completed_epochs:
            lines.extend([
                "",
                "- **Epoch-Level Progress Today**:"
            ])
            for config, epochs in newly_completed_epochs.items():
                config_name = os.path.basename(config)
                epoch_list = ", ".join(map(str, epochs))
                lines.append(f"  - {config_name}: Completed epochs {epoch_list}")
        
        # Add failed experiment details if any
        if failed_experiments:
            lines.extend([
                "",
                "- **Failed Experiments**:"
            ])
            for config, reason in failed_experiments.items():
                config_name = os.path.basename(config)
                lines.append(f"  - {config_name}: {reason}")
        
        return "\\n".join(lines)
    
    def _format_key_events(self, key_events: List[Dict[str, Any]]) -> str:
        """Format key events section.
        
        Args:
            key_events: List of key events from agent log
            
        Returns:
            Formatted key events section
        """
        lines = ["## Key Events"]
        
        if not key_events:
            lines.append("- No significant events occurred today")
            return "\\n".join(lines)
        
        # Group events by type for better organization
        events_by_type = {}
        for event in key_events:
            event_type = event['type']
            if event_type not in events_by_type:
                events_by_type[event_type] = []
            events_by_type[event_type].append(event)
        
        # Format events in priority order
        priority_order = ['critical_error', 'experiment_error', 'stuck_removal', 'outdated_cleanup', 'job_launch']
        
        for event_type in priority_order:
            if event_type in events_by_type:
                for event in events_by_type[event_type]:
                    timestamp = event['timestamp']
                    if isinstance(timestamp, str):
                        time_str = timestamp.split(' ')[1][:5]  # Extract HH:MM from timestamp
                    else:
                        time_str = timestamp.strftime('%H:%M')
                    
                    lines.append(f"- {time_str} - {event['message']}")
        
        return "\\n".join(lines)
    
    def _format_progress_overview(self, current: Dict[str, Any]) -> str:
        """Format progress overview section.
        
        Args:
            current: Current snapshot data
            
        Returns:
            Formatted progress overview section
        """
        current_statuses = current['run_statuses']  # Dict[str, RunStatus]
        
        # Calculate overall completion percentage
        total_progress = 0
        experiment_count = len(current_statuses)
        
        for run_status in current_statuses.values():
            total_progress += run_status['progress']['progress_percentage']
        
        overall_completion = total_progress / experiment_count if experiment_count > 0 else 0
        
        # Find experiments nearing completion (>90%)
        near_completion = []
        for config, run_status in current_statuses.items():
            progress = run_status['progress']['progress_percentage']
            if progress > 90 and run_status['status'] != 'finished':
                near_completion.append((config, progress, run_status['progress']['completed_epochs']))
        
        # Find long-running experiments
        long_running = self._detect_long_running_experiments(current_statuses)
        
        lines = [
            "## Progress Overview",
            f"- **Overall Completion**: {overall_completion:.1f}% across {experiment_count} experiments",
        ]
        
        # Add near completion section
        if near_completion:
            lines.extend([
                "- **Near Completion** (>90%):"
            ])
            for config, progress, completed_epochs in near_completion:
                config_name = os.path.basename(config)
                lines.append(f"  - {config_name}: {progress:.1f}% ({completed_epochs} epochs)")
        
        # Add long running section
        if long_running:
            lines.extend([
                "- **Long Running** (>7 days):"
            ])
            for exp in long_running:
                config_name = os.path.basename(exp['config'])
                lines.append(f"  - {config_name}: Running for {exp['runtime_days']} days ({exp['progress_percentage']:.1f}% complete)")
        
        return "\\n".join(lines)
    
    def _format_resource_utilization(self, current: Dict[str, Any]) -> str:
        """Format resource utilization section.
        
        Args:
            current: Current snapshot data
            
        Returns:
            Formatted resource utilization section
        """
        current_statuses = current['run_statuses']  # Dict[str, RunStatus]
        
        # Count running processes by server
        server_processes = {}
        total_running = 0
        
        for run_status in current_statuses.values():
            if run_status['status'] == 'running' and run_status['process_info']:
                total_running += 1
                # Extract server from process info or default to 'unknown'
                server = 'unknown'
                if 'server' in run_status['process_info']:
                    server = run_status['process_info']['server']
                elif 'cmd' in run_status['process_info']:
                    # Try to extract server from SSH command
                    cmd = run_status['process_info']['cmd']
                    if 'ssh' in cmd and '@' in cmd:
                        try:
                            server = cmd.split('@')[1].split()[0]
                        except (IndexError, AttributeError):
                            pass
                
                server_processes[server] = server_processes.get(server, 0) + 1
        
        lines = [
            "## Resource Utilization",
            f"- **Currently Running Experiments**: {total_running}"
        ]
        
        if server_processes:
            lines.append("- **Distribution by Server**:")
            for server, count in sorted(server_processes.items()):
                lines.append(f"  - {server}: {count} experiment(s)")
        
        # Add GPU utilization note
        lines.extend([
            "",
            "- **Note**: Detailed GPU/CPU utilization metrics require additional monitoring infrastructure"
        ])
        
        return "\\n".join(lines)
    
    def _find_experiments_completed_today(self, current_statuses: Dict[str, Any], previous_statuses: Dict[str, Any]) -> List[str]:
        """Find experiments that completed today (changed from non-finished to finished)."""
        completed_today = []
        
        for config, current_status in current_statuses.items():
            if current_status['status'] == 'finished':
                if config not in previous_statuses:
                    # New experiment that's already finished
                    completed_today.append(config)
                elif previous_statuses[config]['status'] != 'finished':
                    # Status changed to finished
                    completed_today.append(config)
        
        return completed_today
    
    def _find_experiments_started_today(self, current_statuses: Dict[str, Any], previous_statuses: Dict[str, Any]) -> List[str]:
        """Find experiments that started today (new ProcessInfo entries)."""
        started_today = []
        
        for config, current_status in current_statuses.items():
            if current_status['process_info'] and config not in previous_statuses:
                # New experiment with process info
                started_today.append(config)
            elif (current_status['process_info'] and 
                  config in previous_statuses and 
                  not previous_statuses[config]['process_info']):
                # Experiment gained process info (started running)
                started_today.append(config)
        
        return started_today
    
    def _calculate_newly_completed_epochs(self, current_statuses: Dict[str, Any], previous_statuses: Dict[str, Any]) -> Dict[str, List[int]]:
        """Calculate newly completed epochs by comparing snapshots."""
        newly_completed = {}
        
        for config, current_status in current_statuses.items():
            if config in previous_statuses:
                prev_epochs = previous_statuses[config]['progress']['completed_epochs']
                curr_epochs = current_status['progress']['completed_epochs']
                
                if curr_epochs > prev_epochs:
                    newly_completed[config] = list(range(prev_epochs, curr_epochs))
        
        return newly_completed
    
    def _find_failed_experiments(self, current_statuses: Dict[str, Any], previous_statuses: Dict[str, Any]) -> Dict[str, str]:
        """Find experiments that failed and determine failure reasons."""
        failed_experiments = {}
        
        for config, current_status in current_statuses.items():
            if current_status['status'] == 'failed':
                # Determine failure reason based on available information
                reason = "Unknown failure"
                
                # Check if experiment was running before and stopped
                if config in previous_statuses and previous_statuses[config]['status'] == 'running':
                    reason = "Stopped unexpectedly"
                elif current_status['progress']['completed_epochs'] == 0:
                    reason = "Failed to start"
                else:
                    reason = f"Failed after {current_status['progress']['completed_epochs']} epochs"
                
                failed_experiments[config] = reason
        
        return failed_experiments
    
    def _detect_long_running_experiments(self, current_statuses: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect experiments running for more than 7 days."""
        from dateutil import parser
        long_running = []
        
        for config, status in current_statuses.items():
            if status['status'] == 'running' and status['process_info']:
                try:
                    start_time_str = status['process_info']['start_time']
                    start_time = parser.parse(start_time_str)
                    runtime = datetime.now() - start_time
                    
                    if runtime.days > 7:
                        long_running.append({
                            'config': config,
                            'runtime_days': runtime.days,
                            'progress_percentage': status['progress']['progress_percentage'],
                            'server': status['process_info'].get('server', 'unknown')
                        })
                except (ValueError, KeyError, TypeError):
                    # Skip entries with invalid or missing start time
                    continue
        
        return long_running
    
    def save_summary(self, summary: str, date: Optional[str] = None) -> str:
        """Save summary to markdown file.
        
        Args:
            summary: Summary text to save
            date: Date string for filename, defaults to today
            
        Returns:
            Path to saved summary file
        """
        assert isinstance(summary, str), f"summary must be str, got {type(summary)}"
        
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        os.makedirs(self.summary_dir, exist_ok=True)
        filename = f"{date}_summary.md"
        filepath = os.path.join(self.summary_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(summary)
        
        return filepath
    
    def cleanup_old_summaries(self, retention_days: int = 30) -> int:
        """Remove summaries older than retention period.
        
        Args:
            retention_days: Number of days to retain summaries
            
        Returns:
            Number of summaries removed
        """
        assert isinstance(retention_days, int), f"retention_days must be int, got {type(retention_days)}"
        assert retention_days > 0, f"retention_days must be positive, got {retention_days}"
        
        if not os.path.exists(self.summary_dir):
            return 0
        
        import time
        cutoff_time = time.time() - (retention_days * 24 * 60 * 60)
        removed_count = 0
        
        for filename in os.listdir(self.summary_dir):
            if filename.endswith('_summary.md'):
                filepath = os.path.join(self.summary_dir, filename)
                try:
                    if os.path.getmtime(filepath) < cutoff_time:
                        os.remove(filepath)
                        removed_count += 1
                except OSError:
                    continue
        
        return removed_count