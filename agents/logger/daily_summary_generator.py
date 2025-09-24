from typing import List, Dict, Any, Optional
import os
from datetime import datetime
from agents.logger.logs_snapshot import LogsSnapshot
from agents.logger.agent_log_parser import AgentLogParser
from agents.logger.snapshot_diff import SnapshotDiff


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
                              current_snapshot_file: Optional[str] = None,
                              date: Optional[str] = None) -> str:
        """Generate comprehensive daily summary.
        
        Args:
            current_snapshot_file: Path to current snapshot file, defaults to latest
            date: Date string (YYYY-MM-DD) for the summary, defaults to today
            
        Returns:
            Formatted markdown summary string
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # Load current snapshot (latest or specified file)
        if current_snapshot_file is None:
            current_snapshot = self._load_latest_snapshot()
        else:
            current_snapshot = self._load_snapshot_from_file(current_snapshot_file)
        
        # Try to load previous snapshot for comparison
        previous_snapshot = self._load_previous_snapshot()
        
        # Create snapshot diff analyzer
        snapshot_diff = SnapshotDiff(current_snapshot, previous_snapshot)
        
        # Extract key events from agent log
        key_events = self.agent_log_parser.extract_key_events_since_yesterday()
        
        # Generate summary sections in specified order
        previous_snapshot_info = self._get_previous_snapshot_info(previous_snapshot)
        sections = [
            f"# Daily Summary - {date}",
            "",
            f"**Compared to**: {previous_snapshot_info}",
            "",
            self._format_experiment_statistics(current_snapshot, snapshot_diff),
            "",
            self._format_key_events(key_events),
            "",
            self._format_progress_overview(current_snapshot),
            "",
            self._format_resource_utilization(current_snapshot)
        ]
        
        return "\n".join(sections)
    
    def _format_experiment_statistics(self, current: Dict[str, Any], snapshot_diff: SnapshotDiff) -> str:
        """Format experiment statistics section.
        
        Args:
            current: Current snapshot data
            snapshot_diff: SnapshotDiff instance for analyzing changes
            
        Returns:
            Formatted experiment statistics section
        """
        current_statuses = current['job_statuses']  # Dict[str, BaseJob]
        
        # Count experiments by status
        status_counts = {}
        for run_status in current_statuses.values():
            status = run_status['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Calculate experiments completed and started today using SnapshotDiff
        completed_today = snapshot_diff.find_experiments_completed_today()
        started_today = snapshot_diff.find_experiments_started_today()
        
        # Calculate newly completed epochs using SnapshotDiff
        newly_completed_epochs = snapshot_diff.calculate_newly_completed_epochs()
        
        # Find failed experiments using SnapshotDiff
        failed_experiments = snapshot_diff.find_failed_experiments()
        
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
                # Use full config path for clarity
                epoch_list = ", ".join(map(str, epochs))
                lines.append(f"  - {config}: Completed epochs {epoch_list}")
        else:
            lines.extend([
                "",
                "- **Epoch-Level Progress Today**: 0 new epochs completed"
            ])
        
        # Add failed experiment details if any
        if failed_experiments:
            lines.extend([
                "",
                "- **Failed Experiments**:"
            ])
            for config, reason in failed_experiments.items():
                # Use full config path for clarity
                lines.append(f"  - {config}: {reason}")
        else:
            lines.extend([
                "",
                "- **Failed Experiments**: 0 experiments failed today"
            ])
        
        return "\n".join(lines)
    
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
            return "\n".join(lines)
        
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
                    
                    message = f"- {time_str} - {event['message']}"
                    
                    # Add detailed information for specific event types
                    if event_type == 'outdated_cleanup' and 'details' in event:
                        details = event['details']
                        if 'folder_count' in details:
                            message += f" ({details['folder_count']} folders)"
                        if details.get('cleaned_folders'):
                            # Show ALL folders for exhaustive reporting
                            folder_list = ', '.join(details['cleaned_folders'])
                            message += f"\n  **Folders**: {folder_list}"
                    
                    elif event_type == 'stuck_removal' and 'details' in event:
                        details = event['details']
                        if 'process_count' in details:
                            message += f" ({details['process_count']} processes)"
                        if details.get('raw_info'):
                            # Clean up the raw info for better display
                            raw_info = details['raw_info']
                            # Ensure no unclosed brackets in display
                            if raw_info.endswith('...'):
                                # Already truncated, just display
                                message += f"\n  **Details**: {raw_info}"
                            else:
                                message += f"\n  **Details**: {raw_info}"
                        elif details.get('processes'):
                            # If we have parsed processes, show them properly
                            process_list = []
                            for config, proc_info in details['processes'].items():
                                process_list.append(f"{config} on {proc_info['server']} (PID: {proc_info['pid']})")
                            message += f"\n  **Processes**: {', '.join(process_list[:3])}"
                            if len(process_list) > 3:
                                message += f" and {len(process_list) - 3} more"
                    
                    lines.append(message)
        
        return "\n".join(lines)
    
    def _format_progress_overview(self, current: Dict[str, Any]) -> str:
        """Format progress overview section.
        
        Args:
            current: Current snapshot data
            
        Returns:
            Formatted progress overview section
        """
        current_statuses = current['job_statuses']  # Dict[str, BaseJob]
        
        # Calculate overall completion percentage
        total_progress = 0
        experiment_count = len(current_statuses)
        
        for run_status in current_statuses.values():
            if run_status.get('progress') and run_status['progress'].get('progress_percentage') is not None:
                total_progress += run_status['progress']['progress_percentage']
        
        overall_completion = total_progress / experiment_count if experiment_count > 0 else 0
        
        # Find experiments nearing completion (>90%)
        near_completion = []
        for config, run_status in current_statuses.items():
            if run_status.get('progress') and run_status['progress'].get('progress_percentage') is not None:
                progress = run_status['progress']['progress_percentage']
                if progress > 90 and run_status.get('status') != 'finished':
                    completed_epochs = run_status['progress'].get('completed_epochs', 0)
                    near_completion.append((config, progress, completed_epochs))
        
        # Find long-running experiments
        long_running = self._detect_long_running_experiments(current_statuses)
        
        lines = [
            "## Progress Overview",
            f"- **Overall Completion**: {overall_completion:.1f}% across {experiment_count} experiments",
        ]
        
        # Add near completion section
        if near_completion:
            lines.extend([
                f"- **Near Completion** (>90%): {len(near_completion)} experiments"
            ])
            # Show ALL experiments with full paths - exhaustive reporting
            for config, progress, completed_epochs in near_completion:
                lines.append(f"  - {config}: {progress:.1f}% ({completed_epochs} epochs)")
        else:
            lines.append("- **Near Completion** (>90%): 0 experiments")
        
        # Add long running section
        if long_running:
            lines.extend([
                "- **Long Running** (>7 days):"
            ])
            for exp in long_running:
                # Use full config path for clarity
                lines.append(f"  - {exp['config']}: Running for {exp['runtime_days']} days ({exp['progress_percentage']:.1f}% complete)")
        else:
            lines.append("- **Long Running** (>7 days): 0 experiments")
        
        return "\n".join(lines)
    
    def _format_resource_utilization(self, current: Dict[str, Any]) -> str:
        """Format resource utilization section.
        
        Args:
            current: Current snapshot data
            
        Returns:
            Formatted resource utilization section
        """
        current_statuses = current['job_statuses']  # Dict[str, BaseJob]
        
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
        
        return "\n".join(lines)
    
    
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
    
    def _load_latest_snapshot(self) -> Dict[str, Any]:
        """Load the most recent snapshot file."""
        import glob
        
        snapshot_dir = "./agents/snapshots"
        if not os.path.exists(snapshot_dir):
            return {'job_statuses': {}, 'snapshot_metadata': {'total_configs': 0}}
        
        # Find all snapshot files
        snapshot_files = glob.glob(os.path.join(snapshot_dir, "*.json"))
        if not snapshot_files:
            return {'job_statuses': {}, 'snapshot_metadata': {'total_configs': 0}}
        
        # Get the most recent file
        latest_file = max(snapshot_files, key=os.path.getmtime)
        return self._load_snapshot_from_file(latest_file)
    
    def _load_snapshot_from_file(self, filepath: str) -> Dict[str, Any]:
        """Load snapshot from specific file."""
        try:
            import json
            with open(filepath, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {'job_statuses': {}, 'snapshot_metadata': {'total_configs': 0}}
    
    def _load_previous_snapshot(self) -> Optional[Dict[str, Any]]:
        """Load previous snapshot for comparison."""
        import glob
        
        snapshot_dir = "./agents/snapshots"
        if not os.path.exists(snapshot_dir):
            return None
        
        # Find all snapshot files
        snapshot_files = glob.glob(os.path.join(snapshot_dir, "*.json"))
        if len(snapshot_files) < 2:
            return None
        
        # Get the second most recent file
        sorted_files = sorted(snapshot_files, key=os.path.getmtime, reverse=True)
        previous_file = sorted_files[1]
        
        try:
            import json
            with open(previous_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
    
    def _get_previous_snapshot_info(self, previous_snapshot: Optional[Dict[str, Any]]) -> str:
        """Get information about the previous snapshot being compared to."""
        if previous_snapshot is None:
            return "No previous snapshot available"
        
        timestamp = previous_snapshot.get('timestamp', 'Unknown')
        total_configs = previous_snapshot.get('snapshot_metadata', {}).get('total_configs', 0)
        active_experiments = len(previous_snapshot.get('job_statuses', {}))
        
        return f"Snapshot {timestamp} ({active_experiments}/{total_configs} active experiments)"
