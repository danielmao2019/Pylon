from typing import List, Dict, Any, Optional
import os
import json
from datetime import datetime
from utils.monitor.system_monitor import SystemMonitor
from agents.manager import get_all_job_status
from utils.io.json import serialize_object


class LogsSnapshot:
    """Creates and manages snapshots of logs directory state.
    
    This class provides a simple wrapper around the enhanced run_status functionality
    to create daily snapshots of experiment states for comparison and analysis.
    """
    
    def __init__(self, 
                 config_files: List[str],
                 epochs: int,
                 sleep_time: int = 86400,
                 outdated_days: int = 30):
        """Initialize snapshot manager.
        
        Args:
            config_files: List of config file paths to monitor
            epochs: Total number of epochs expected for experiments
            sleep_time: Time to wait for status updates (seconds)
            outdated_days: Number of days to consider a run outdated
        """
        assert isinstance(config_files, list), f"config_files must be list, got {type(config_files)}"
        assert isinstance(epochs, int), f"epochs must be int, got {type(epochs)}"
        assert isinstance(sleep_time, int), f"sleep_time must be int, got {type(sleep_time)}"
        assert isinstance(outdated_days, int), f"outdated_days must be int, got {type(outdated_days)}"
        
        self.config_files = config_files
        self.epochs = epochs
        self.sleep_time = sleep_time
        self.outdated_days = outdated_days
        self.snapshot_dir = "./agents/snapshots"
        
    def create_snapshot(self, timestamp: str, system_monitor: SystemMonitor) -> Dict[str, Any]:
        """Create snapshot of logs directory state using enhanced run_status.
        
        Args:
            timestamp: Timestamp string for the snapshot
            system_monitor: SystemMonitor instance for GPU/process queries
            
        Returns:
            Dictionary containing snapshot data with enhanced RunStatus mapping
        """
        assert isinstance(timestamp, str), f"timestamp must be str, got {type(timestamp)}"
        assert isinstance(system_monitor, SystemMonitor), f"system_monitor must be SystemMonitor, got {type(system_monitor)}"
        
        # Use enhanced get_all_run_status that returns Dict[str, RunStatus] with ProcessInfo
        run_statuses = get_all_job_status(
            config_files=self.config_files,
            epochs=self.epochs,
            sleep_time=self.sleep_time,
            outdated_days=self.outdated_days,
            system_monitor=system_monitor
        )
        
        snapshot = {
            'timestamp': timestamp,
            'run_statuses': run_statuses,  # Dict[str, RunStatus] with enhanced progress and ProcessInfo
            'snapshot_metadata': {
                'total_configs': len(self.config_files),
                'epochs': self.epochs,
                'sleep_time': self.sleep_time,
                'outdated_days': self.outdated_days
            }
        }
        
        return snapshot
        
    def save_snapshot(self, snapshot: Dict[str, Any], filename: str) -> None:
        """Save snapshot to JSON file.
        
        Args:
            snapshot: Snapshot data dictionary
            filename: Filename to save snapshot as
        """
        assert isinstance(snapshot, dict), f"snapshot must be dict, got {type(snapshot)}"
        assert isinstance(filename, str), f"filename must be str, got {type(filename)}"
        assert filename.endswith('.json'), f"filename must end with .json, got {filename}"
        
        os.makedirs(self.snapshot_dir, exist_ok=True)
        filepath = os.path.join(self.snapshot_dir, filename)
        
        # Convert snapshot to JSON-serializable format using generic serializer
        serializable_snapshot = serialize_object(snapshot)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_snapshot, f, indent=2)
            
    def load_snapshot(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load snapshot from JSON file.
        
        Args:
            filename: Filename to load snapshot from
            
        Returns:
            Snapshot data dictionary if file exists, None otherwise
        """
        assert isinstance(filename, str), f"filename must be str, got {type(filename)}"
        
        filepath = os.path.join(self.snapshot_dir, filename)
        if not os.path.exists(filepath):
            return None
            
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            # Log error but don't crash - return None for missing/corrupted snapshots
            print(f"Warning: Failed to load snapshot {filename}: {str(e)}")
            return None
    
    def list_snapshots(self) -> List[str]:
        """List all available snapshot files.
        
        Returns:
            List of snapshot filenames sorted by creation time
        """
        if not os.path.exists(self.snapshot_dir):
            return []
            
        snapshot_files = [
            f for f in os.listdir(self.snapshot_dir) 
            if f.endswith('.json')
        ]
        
        # Sort by modification time (newest first)
        snapshot_files.sort(
            key=lambda f: os.path.getmtime(os.path.join(self.snapshot_dir, f)),
            reverse=True
        )
        
        return snapshot_files
    
    def cleanup_old_snapshots(self, retention_days: int = 30) -> int:
        """Remove snapshots older than retention period.
        
        Args:
            retention_days: Number of days to retain snapshots
            
        Returns:
            Number of snapshots removed
        """
        assert isinstance(retention_days, int), f"retention_days must be int, got {type(retention_days)}"
        assert retention_days > 0, f"retention_days must be positive, got {retention_days}"
        
        if not os.path.exists(self.snapshot_dir):
            return 0
            
        import time
        cutoff_time = time.time() - (retention_days * 24 * 60 * 60)
        removed_count = 0
        
        for filename in os.listdir(self.snapshot_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.snapshot_dir, filename)
                if os.path.getmtime(filepath) < cutoff_time:
                    try:
                        os.remove(filepath)
                        removed_count += 1
                    except OSError:
                        # Skip files that can't be removed
                        continue
                        
        return removed_count
    
    def get_snapshot_statistics(self) -> Dict[str, Any]:
        """Get statistics about available snapshots.
        
        Returns:
            Dictionary with snapshot statistics
        """
        snapshots = self.list_snapshots()
        
        if not snapshots:
            return {
                'total_snapshots': 0,
                'oldest_snapshot': None,
                'newest_snapshot': None,
                'total_size_bytes': 0
            }
        
        total_size = 0
        oldest_time = float('inf')
        newest_time = 0
        
        for filename in snapshots:
            filepath = os.path.join(self.snapshot_dir, filename)
            try:
                stat = os.stat(filepath)
                total_size += stat.st_size
                oldest_time = min(oldest_time, stat.st_mtime)
                newest_time = max(newest_time, stat.st_mtime)
            except OSError:
                continue
        
        return {
            'total_snapshots': len(snapshots),
            'oldest_snapshot': datetime.fromtimestamp(oldest_time).isoformat() if oldest_time != float('inf') else None,
            'newest_snapshot': datetime.fromtimestamp(newest_time).isoformat() if newest_time > 0 else None,
            'total_size_bytes': total_size
        }
