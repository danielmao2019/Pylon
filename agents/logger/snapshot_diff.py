from typing import Dict, List, Optional, Any


class SnapshotDiff:
    """Analyzes differences between two experiment snapshots.

    Provides methods to identify changes in experiment status, progress,
    and other metrics between snapshots taken at different times.
    """

    def __init__(self, current_snapshot: Dict[str, Any], previous_snapshot: Optional[Dict[str, Any]] = None):
        """Initialize snapshot diff analyzer.

        Args:
            current_snapshot: Current snapshot data
            previous_snapshot: Previous snapshot data for comparison (may be None)
        """
        assert isinstance(current_snapshot, dict), f"current_snapshot must be dict, got {type(current_snapshot)}"
        assert 'job_statuses' in current_snapshot, "current_snapshot must have 'job_statuses' key"

        if previous_snapshot is not None:
            assert isinstance(previous_snapshot, dict), f"previous_snapshot must be dict, got {type(previous_snapshot)}"
            assert 'job_statuses' in previous_snapshot, "previous_snapshot must have 'job_statuses' key"

        self.current_snapshot = current_snapshot
        self.previous_snapshot = previous_snapshot
        self.current_statuses = current_snapshot['job_statuses']
        self.previous_statuses = previous_snapshot['job_statuses'] if previous_snapshot else {}

    @classmethod
    def create_diff(cls, current_snapshot: Dict[str, Any], previous_snapshot: Optional[Dict[str, Any]] = None) -> 'SnapshotDiff':
        """Factory method to create a snapshot diff.

        Args:
            current_snapshot: Current snapshot data
            previous_snapshot: Previous snapshot data for comparison

        Returns:
            SnapshotDiff instance for analyzing changes
        """
        return cls(current_snapshot, previous_snapshot)

    def find_experiments_completed_today(self) -> List[str]:
        """Find experiments that completed today (changed from non-finished to finished).

        Returns:
            List of config paths for experiments that completed today
        """
        completed_today = []

        for config, current_status in self.current_statuses.items():
            if current_status['status'] == 'finished':
                if config not in self.previous_statuses:
                    # New experiment that's already finished
                    completed_today.append(config)
                elif self.previous_statuses[config]['status'] != 'finished':
                    # Status changed to finished
                    completed_today.append(config)

        return completed_today

    def find_experiments_started_today(self) -> List[str]:
        """Find experiments that started today (new ProcessInfo entries).

        Returns:
            List of config paths for experiments that started today
        """
        started_today = []

        for config, current_status in self.current_statuses.items():
            if current_status['process_info'] and config not in self.previous_statuses:
                # New experiment with process info
                started_today.append(config)
            elif (current_status['process_info'] and
                  config in self.previous_statuses and
                  not self.previous_statuses[config]['process_info']):
                # Experiment gained process info (started running)
                started_today.append(config)

        return started_today

    def calculate_newly_completed_epochs(self) -> Dict[str, List[int]]:
        """Calculate newly completed epochs by comparing snapshots.

        Returns:
            Dictionary mapping config paths to lists of newly completed epoch numbers
        """
        newly_completed = {}

        for config, current_status in self.current_statuses.items():
            if config in self.previous_statuses:
                prev_epochs = self.previous_statuses[config]['progress']['completed_epochs']
                curr_epochs = current_status['progress']['completed_epochs']

                if curr_epochs > prev_epochs:
                    newly_completed[config] = list(range(prev_epochs, curr_epochs))

        return newly_completed

    def find_failed_experiments(self) -> Dict[str, str]:
        """Find experiments that failed and determine failure reasons.

        Returns:
            Dictionary mapping config paths to failure reason strings
        """
        failed_experiments = {}

        for config, current_status in self.current_statuses.items():
            if current_status['status'] == 'failed':
                # Determine failure reason based on available information
                reason = "Unknown failure"

                # Check if experiment was running before and stopped
                if config in self.previous_statuses and self.previous_statuses[config]['status'] == 'running':
                    reason = "Stopped unexpectedly"
                elif current_status['progress']['completed_epochs'] == 0:
                    reason = "Failed to start"
                else:
                    reason = f"Failed after {current_status['progress']['completed_epochs']} epochs"

                failed_experiments[config] = reason

        return failed_experiments

    def find_status_changes(self) -> Dict[str, Dict[str, str]]:
        """Find all experiments that changed status between snapshots.

        Returns:
            Dictionary mapping config paths to status change info:
            {'previous_status': str, 'current_status': str}
        """
        status_changes = {}

        for config, current_status in self.current_statuses.items():
            current_state = current_status['status']

            if config in self.previous_statuses:
                previous_state = self.previous_statuses[config]['status']
                if previous_state != current_state:
                    status_changes[config] = {
                        'previous_status': previous_state,
                        'current_status': current_state
                    }
            else:
                # New experiment
                status_changes[config] = {
                    'previous_status': 'not_present',
                    'current_status': current_state
                }

        return status_changes

    def find_progress_changes(self) -> Dict[str, Dict[str, Any]]:
        """Find experiments with significant progress changes.

        Returns:
            Dictionary mapping config paths to progress change info:
            {'previous_epochs': int, 'current_epochs': int, 'epochs_gained': int}
        """
        progress_changes = {}

        for config, current_status in self.current_statuses.items():
            if config in self.previous_statuses:
                prev_epochs = self.previous_statuses[config]['progress']['completed_epochs']
                curr_epochs = current_status['progress']['completed_epochs']

                if curr_epochs != prev_epochs:
                    progress_changes[config] = {
                        'previous_epochs': prev_epochs,
                        'current_epochs': curr_epochs,
                        'epochs_gained': curr_epochs - prev_epochs
                    }

        return progress_changes

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics about changes between snapshots.

        Returns:
            Dictionary with counts of various types of changes
        """
        completed_today = self.find_experiments_completed_today()
        started_today = self.find_experiments_started_today()
        failed_experiments = self.find_failed_experiments()
        status_changes = self.find_status_changes()
        newly_completed_epochs = self.calculate_newly_completed_epochs()

        total_new_epochs = sum(len(epochs) for epochs in newly_completed_epochs.values())

        return {
            'experiments_completed_today': len(completed_today),
            'experiments_started_today': len(started_today),
            'experiments_failed_today': len(failed_experiments),
            'experiments_with_status_changes': len(status_changes),
            'experiments_with_epoch_progress': len(newly_completed_epochs),
            'total_new_epochs_completed': total_new_epochs,
            'has_previous_snapshot': self.previous_snapshot is not None,
            'current_active_experiments': len(self.current_statuses),
            'previous_active_experiments': len(self.previous_statuses) if self.previous_snapshot else 0
        }

    def get_snapshot_metadata_comparison(self) -> Dict[str, Any]:
        """Compare metadata between current and previous snapshots.

        Returns:
            Dictionary with metadata comparison information
        """
        current_meta = self.current_snapshot.get('snapshot_metadata', {})
        previous_meta = self.previous_snapshot.get('snapshot_metadata', {}) if self.previous_snapshot else {}

        current_timestamp = self.current_snapshot.get('timestamp', 'Unknown')
        previous_timestamp = self.previous_snapshot.get('timestamp', 'Unknown') if self.previous_snapshot else 'No previous snapshot'

        return {
            'current_timestamp': current_timestamp,
            'previous_timestamp': previous_timestamp,
            'current_total_configs': current_meta.get('total_configs', 0),
            'previous_total_configs': previous_meta.get('total_configs', 0),
            'current_active_experiments': len(self.current_statuses),
            'previous_active_experiments': len(self.previous_statuses)
        }
