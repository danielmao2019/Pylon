from typing import List, Dict, Optional, Any
import os
import re
from datetime import datetime, timedelta


class AgentLogParser:
    """Parser for extracting key events from agent log files.

    Parses structured log entries to extract important events like:
    - Stuck process removal
    - Outdated cleanup operations
    - Critical errors
    - Job launching events
    """

    def __init__(self, agent_log_path: str = "./project/run_agent.log"):
        """Initialize parser with agent log file path.

        Args:
            agent_log_path: Path to the agent log file
        """
        self.agent_log_path = agent_log_path

        # Compile regex patterns for efficient parsing
        self.patterns = {
            'stuck_removal': re.compile(
                r'The following processes will be killed.*?\{([^}]+)\}'
            ),
            'outdated_cleanup': re.compile(
                r'The following runs has not been updated in the last (\d+) days and will be removed'
            ),
            'error_message': re.compile(
                r'Please fix ([^.]+\.py)\. error_log=\'([^\']+)\''
            ),
            'ssh_launch': re.compile(
                r'ssh.*?python main\.py --config-filepath ([^\s\']+)'
            ),
            'timestamp': re.compile(
                r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
            )
        }

    def extract_key_events_since_yesterday(self) -> List[Dict[str, Any]]:
        """Extract key events that occurred since yesterday.

        Returns:
            List of event dictionaries with standardized structure
        """
        if not os.path.exists(self.agent_log_path):
            return []

        yesterday = datetime.now() - timedelta(days=1)
        events = []

        with open(self.agent_log_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                event = self._parse_log_line(line.strip(), line_num)
                if event and event['timestamp'] >= yesterday:
                    events.append(event)

        return sorted(events, key=lambda x: x['timestamp'])

    def _parse_log_line(self, line: str, line_num: int) -> Optional[Dict[str, Any]]:
        """Parse a single log line and extract event information.

        Args:
            line: Log line content
            line_num: Line number for debugging

        Returns:
            Event dictionary if line contains a key event, None otherwise
        """
        # Extract timestamp from line
        timestamp_match = self.patterns['timestamp'].search(line)
        if not timestamp_match:
            return None

        timestamp = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')

        # Check for stuck process removal
        stuck_match = self.patterns['stuck_removal'].search(line)
        if stuck_match:
            return {
                'type': 'stuck_removal',
                'timestamp': timestamp,
                'message': f"Stuck processes removed: {stuck_match.group(1)}",
                'details': self._parse_stuck_processes(stuck_match.group(1)),
                'line_number': line_num
            }

        # Check for outdated cleanup
        outdated_match = self.patterns['outdated_cleanup'].search(line)
        if outdated_match:
            days = int(outdated_match.group(1))
            # Try to extract folder list from next lines
            cleaned_folders = self._extract_cleaned_folders(line)
            return {
                'type': 'outdated_cleanup',
                'timestamp': timestamp,
                'message': f"Cleaned outdated runs (>{days} days old)",
                'details': {
                    'days_threshold': days,
                    'cleaned_folders': cleaned_folders,
                    'folder_count': len(cleaned_folders)
                },
                'line_number': line_num
            }

        # Check for error messages
        error_match = self.patterns['error_message'].search(line)
        if error_match:
            config_file = error_match.group(1)
            error_log = error_match.group(2)
            return {
                'type': 'experiment_error',
                'timestamp': timestamp,
                'message': f"Experiment failed: {config_file}",
                'details': {
                    'config_file': config_file,
                    'error_log': error_log[:200] + '...' if len(error_log) > 200 else error_log
                },
                'line_number': line_num
            }

        # Check for SSH job launches
        ssh_match = self.patterns['ssh_launch'].search(line)
        if ssh_match:
            config_path = ssh_match.group(1)
            return {
                'type': 'job_launch',
                'timestamp': timestamp,
                'message': f"Launched experiment: {os.path.basename(config_path)}",
                'details': {'config_path': config_path},
                'line_number': line_num
            }

        # Check for critical system errors (SSH failures, crashes, etc.)
        critical_keywords = ['ssh.*connection failed', 'SystemMonitor.*disconnected', 'Agent.*crash', 'disk.*full', 'permission.*denied']
        for keyword_pattern in critical_keywords:
            if re.search(keyword_pattern, line, re.IGNORECASE):
                return {
                    'type': 'critical_error',
                    'timestamp': timestamp,
                    'message': f"Critical system error detected",
                    'details': {'error_line': line[:300] + '...' if len(line) > 300 else line},
                    'line_number': line_num
                }

        return None

    def _parse_stuck_processes(self, processes_str: str) -> Dict[str, Any]:
        """Parse stuck processes information from log entry.

        Args:
            processes_str: String containing process information

        Returns:
            Dictionary with parsed process details
        """
        # Parse format like: "config1.py: (server1, 12345), config2.py: (server2, 67890)"
        processes = {}

        # First split by commas that are NOT inside parentheses
        parts = []
        current_part = ""
        paren_count = 0

        for char in processes_str:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == ',' and paren_count == 0:
                parts.append(current_part.strip())
                current_part = ""
                continue
            current_part += char

        if current_part.strip():
            parts.append(current_part.strip())

        for part in parts:
            if ':' in part:
                config_part, server_pid_part = part.split(':', 1)
                config = config_part.strip()

                # Extract server and PID from format like "(server1, 12345)"
                server_pid_match = re.search(r'\(([^,]+),\s*(\d+)\)', server_pid_part)
                if server_pid_match:
                    server = server_pid_match.group(1).strip()
                    pid = server_pid_match.group(2).strip()
                    processes[config] = {'server': server, 'pid': pid}

        # If no processes were parsed successfully, return fallback info
        if len(processes) == 0:
            # For exhaustive reporting, return the full information without truncation
            return {
                'process_count': 0,
                'raw_info': processes_str
            }

        return {
            'process_count': len(processes),
            'processes': processes
        }

    def get_event_summary_since_yesterday(self) -> Dict[str, Any]:
        """Get a summary of events that occurred since yesterday.

        Returns:
            Dictionary with event counts and key statistics
        """
        events = self.extract_key_events_since_yesterday()

        summary = {
            'total_events': len(events),
            'event_counts': {},
            'critical_issues': 0,
            'experiments_affected': set(),
            'time_range': {
                'start': None,
                'end': None
            }
        }

        if not events:
            summary['experiments_affected'] = 0
            return summary

        # Count events by type
        for event in events:
            event_type = event['type']
            summary['event_counts'][event_type] = summary['event_counts'].get(event_type, 0) + 1

            # Track critical issues
            if event_type in ['critical_error', 'experiment_error']:
                summary['critical_issues'] += 1

            # Track affected experiments
            if event_type in ['stuck_removal', 'experiment_error', 'job_launch']:
                if 'config_path' in event.get('details', {}):
                    summary['experiments_affected'].add(event['details']['config_path'])
                elif 'config_file' in event.get('details', {}):
                    summary['experiments_affected'].add(event['details']['config_file'])

        # Set time range
        summary['time_range']['start'] = events[0]['timestamp']
        summary['time_range']['end'] = events[-1]['timestamp']
        summary['experiments_affected'] = len(summary['experiments_affected'])

        return summary

    def _extract_cleaned_folders(self, line: str) -> List[str]:
        """Extract list of cleaned folders from log line.

        Args:
            line: Log line that may contain folder information

        Returns:
            List of folder paths that were cleaned
        """
        # For now, return empty list - would need to parse actual log format
        # This could be enhanced to read subsequent lines or parse specific patterns
        folders = []

        # Try to extract folder paths if they're in the same line
        folder_patterns = [
            r'./logs/[^,\s]+',  # Match log folder paths
            r'/[^,\s]+/logs/[^,\s]+',  # Match absolute log paths
        ]

        for pattern in folder_patterns:
            matches = re.findall(pattern, line)
            folders.extend(matches)

        return folders  # Return all folders for exhaustive reporting
