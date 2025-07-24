from typing import Dict, Any, Optional, Literal, List
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import time
import os
from utils.io.json import safe_save_json


@dataclass  
class ProgressInfo:
    """Simplified progress information focusing on progress tracking only."""
    # REQUIRED: Existing fields (backward compatibility)
    completed_epochs: int
    progress_percentage: float
    early_stopped: bool = False
    early_stopped_at_epoch: Optional[int] = None
    
    # NEW: Runner type identification
    runner_type: Literal['trainer', 'evaluator', 'multi_stage'] = 'trainer'
    
    # NEW: Enhanced metadata (but focused)
    total_epochs: Optional[int] = None  # None for evaluators
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization (backward compatibility)."""
        return asdict(self)


class BaseProgressTracker(ABC):
    """Abstract base class for progress tracking."""
    
    def __init__(self, work_dir: str, config: Optional[Dict[str, Any]] = None):
        assert isinstance(work_dir, str), f"work_dir must be str, got {type(work_dir)}"
        assert os.path.exists(work_dir), f"work_dir does not exist: {work_dir}"
        
        self.work_dir = work_dir  
        self.config = config
        self._cache: Optional[ProgressInfo] = None
        self._cache_time: Optional[float] = None
        self._cache_timeout: float = 5.0  # Cache for 5 seconds
    
    @abstractmethod
    def get_runner_type(self) -> Literal['trainer', 'evaluator', 'multi_stage']:
        """Return the type of runner this tracker handles."""
        pass
    
    @abstractmethod 
    def get_expected_files(self) -> List[str]:
        """Return list of expected files for completion checking."""
        pass
    
    @abstractmethod
    def get_log_pattern(self) -> str:
        """Return the log file pattern to check for activity."""
        pass
    
    @abstractmethod
    def calculate_progress(self) -> ProgressInfo:
        """Calculate and return current progress information.""" 
        pass
    
    
    def get_progress(self, force_refresh: bool = False) -> ProgressInfo:
        """Get progress info with caching support."""
        if not force_refresh and self._cache and self._should_use_cache():
            return self._cache
        
        progress = self.calculate_progress()
        self._cache = progress
        self._cache_time = time.time()
        
        # Save to progress.json for fast access
        self._save_progress_cache(progress)
        return progress
    
    def _should_use_cache(self) -> bool:
        """Check if cached progress is still valid."""
        if self._cache_time is None:
            return False
        return (time.time() - self._cache_time) < self._cache_timeout
    
    def _save_progress_cache(self, progress: ProgressInfo) -> None:
        """Save progress info to progress.json for fast subsequent access."""
        progress_file = os.path.join(self.work_dir, "progress.json")
        safe_save_json(progress.to_dict(), progress_file)
