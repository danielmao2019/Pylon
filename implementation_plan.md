# Generalized Progress Tracking Implementation Plan

## Overview

This document outlines the plan to generalize Pylon's progress tracking functionality to support both BaseTrainer and BaseEvaluator results. Currently, the progress tracking is tailored specifically for trainer results with epoch-based directory structures. This plan will create a unified system that can handle different runner types while maintaining backward compatibility.

## Current State Analysis

### Trainer Progress Tracking
- **Structure**: Epoch-based directories (`epoch_0/`, `epoch_1/`, etc.)
- **Expected Files**: `["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]`
- **Progress Calculation**: `completed_epochs / total_epochs * 100`
- **Log Pattern**: `train_val*.log`
- **Status**: running, finished, stuck, failed, outdated

### Evaluator Progress Tracking  
- **Structure**: Flat directory (no epochs)
- **Expected Files**: `["evaluation_scores.json"]`
- **Progress**: Binary (not started, running, finished)
- **Log Pattern**: `eval_*.log`
- **No epoch-based progress tracking currently implemented**

### Eval Viewer Patterns (Already Handles Both)
- **Runner Type Detection**: Checks for specific file patterns
- **Shape Handling**: `(N, C, H, W)` for trainer, `(C, H, W)` for evaluator
- **Unified Display**: Single interface for both result types

## Design Goals

1. **Unified Interface**: Single API that works for both trainers and evaluators
2. **Backward Compatibility**: Existing trainer-based code continues to work
3. **Extensibility**: Easy to add new runner types (e.g., multi-stage training)
4. **Performance**: Efficient progress checking with caching
5. **Consistency**: Align with eval_viewer's existing patterns

## Architecture Design

### 1. Abstract Base Class

```python
# utils/automation/run_status/base_progress_tracker.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Literal
from dataclasses import dataclass

@dataclass
class ProgressInfo:
    """Unified progress information structure."""
    runner_type: Literal['trainer', 'evaluator', 'multi_stage']
    work_dir: str
    status: Literal['not_started', 'running', 'finished', 'stuck', 'failed', 'outdated']
    progress_percentage: float
    
    # Common fields
    start_time: Optional[float] = None
    last_update_time: Optional[float] = None
    expected_duration: Optional[float] = None
    
    # Trainer-specific fields
    completed_epochs: Optional[int] = None
    total_epochs: Optional[int] = None
    early_stopped: bool = False
    early_stopped_at_epoch: Optional[int] = None
    
    # Score tracking
    best_score: Optional[Dict[str, float]] = None
    latest_score: Optional[Dict[str, float]] = None
    
    # Additional metadata
    error_message: Optional[str] = None
    warnings: Optional[List[str]] = None

class BaseProgressTracker(ABC):
    """Abstract base class for progress tracking."""
    
    def __init__(self, work_dir: str, config: Dict[str, Any]):
        self.work_dir = work_dir
        self.config = config
        self._cache: Optional[ProgressInfo] = None
        self._cache_time: Optional[float] = None
    
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
    
    @abstractmethod
    def is_complete(self) -> bool:
        """Check if the run is complete."""
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
```

### 2. Trainer Progress Tracker

```python
# utils/automation/run_status/trainer_progress_tracker.py
class TrainerProgressTracker(BaseProgressTracker):
    """Progress tracker for BaseTrainer runs."""
    
    def get_runner_type(self) -> Literal['trainer']:
        return 'trainer'
    
    def get_expected_files(self) -> List[str]:
        return ["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]
    
    def get_log_pattern(self) -> str:
        return "train_val*.log"
    
    def calculate_progress(self) -> ProgressInfo:
        """Calculate trainer-specific progress."""
        # Check epoch directories
        completed_epochs = 0
        total_epochs = self.config.get('epochs', 0)
        
        for epoch_idx in range(total_epochs):
            epoch_dir = os.path.join(self.work_dir, f"epoch_{epoch_idx}")
            if check_epoch_finished(epoch_dir, self.get_expected_files()):
                completed_epochs += 1
            else:
                break
        
        # Check for early stopping
        early_stopped, early_stopped_at = self._check_early_stopping()
        
        # Get latest scores
        best_score, latest_score = self._get_scores(completed_epochs)
        
        # Determine status
        status = self._determine_status(completed_epochs, total_epochs)
        
        return ProgressInfo(
            runner_type='trainer',
            work_dir=self.work_dir,
            status=status,
            progress_percentage=(completed_epochs / total_epochs * 100) if total_epochs > 0 else 0,
            completed_epochs=completed_epochs,
            total_epochs=total_epochs,
            early_stopped=early_stopped,
            early_stopped_at_epoch=early_stopped_at,
            best_score=best_score,
            latest_score=latest_score,
            # ... other fields
        )
```

### 3. Evaluator Progress Tracker

```python
# utils/automation/run_status/evaluator_progress_tracker.py
class EvaluatorProgressTracker(BaseProgressTracker):
    """Progress tracker for BaseEvaluator runs."""
    
    def get_runner_type(self) -> Literal['evaluator']:
        return 'evaluator'
    
    def get_expected_files(self) -> List[str]:
        return ["evaluation_scores.json"]
    
    def get_log_pattern(self) -> str:
        return "eval_*.log"
    
    def calculate_progress(self) -> ProgressInfo:
        """Calculate evaluator-specific progress."""
        # Check if evaluation is complete
        eval_complete = all(
            os.path.exists(os.path.join(self.work_dir, f)) 
            for f in self.get_expected_files()
        )
        
        # Check if running
        is_running = self._check_if_running()
        
        # Determine status and progress
        if eval_complete:
            status = 'finished'
            progress = 100.0
        elif is_running:
            status = 'running'
            progress = 50.0  # Arbitrary mid-point
        else:
            status = 'not_started'
            progress = 0.0
        
        # Get evaluation scores if available
        scores = self._get_evaluation_scores() if eval_complete else None
        
        return ProgressInfo(
            runner_type='evaluator',
            work_dir=self.work_dir,
            status=status,
            progress_percentage=progress,
            latest_score=scores,
            best_score=scores,  # Same as latest for single evaluation
            # ... other fields
        )
```

### 4. Progress Tracker Factory

```python
# utils/automation/run_status/progress_tracker_factory.py
def detect_runner_type(work_dir: str, config: Optional[Dict[str, Any]] = None) -> Literal['trainer', 'evaluator']:
    """Detect runner type from work_dir structure or config."""
    # First try to detect from existing files (like eval_viewer does)
    if os.path.exists(os.path.join(work_dir, "evaluation_scores.json")):
        return 'evaluator'
    
    if os.path.exists(os.path.join(work_dir, "epoch_0")):
        return 'trainer'
    
    # Then check config if available
    if config:
        runner_class = config.get('runner', {}).get('class', None)
        if runner_class:
            class_name = runner_class.__name__ if hasattr(runner_class, '__name__') else str(runner_class)
            if 'Evaluator' in class_name:
                return 'evaluator'
            elif 'Trainer' in class_name:
                return 'trainer'
    
    # Default to trainer for backward compatibility
    return 'trainer'

def create_progress_tracker(work_dir: str, config: Dict[str, Any]) -> BaseProgressTracker:
    """Factory function to create appropriate progress tracker."""
    runner_type = detect_runner_type(work_dir, config)
    
    if runner_type == 'evaluator':
        return EvaluatorProgressTracker(work_dir, config)
    else:
        return TrainerProgressTracker(work_dir, config)
```

### 5. Updated Run Status Module

```python
# utils/automation/run_status/run_status.py
# Update existing functions to use progress trackers

def get_run_status(work_dir: str, config: Optional[Dict[str, Any]] = None, 
                   sleep_time: int = 86400, outdated_days: int = 3) -> str:
    """Get run status using appropriate progress tracker."""
    # Create progress tracker
    tracker = create_progress_tracker(work_dir, config)
    progress = tracker.get_progress()
    
    # The status is now directly available from ProgressInfo
    return progress.status

def get_run_progress(work_dir: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get detailed progress information."""
    tracker = create_progress_tracker(work_dir, config)
    progress = tracker.get_progress()
    
    # Convert to dict for backward compatibility
    return {
        'runner_type': progress.runner_type,
        'status': progress.status,
        'progress_percentage': progress.progress_percentage,
        'completed_epochs': progress.completed_epochs,
        'total_epochs': progress.total_epochs,
        'early_stopped': progress.early_stopped,
        'early_stopped_at_epoch': progress.early_stopped_at_epoch,
        'best_score': progress.best_score,
        'latest_score': progress.latest_score,
    }
```

### 6. Updated Agent Module

```python
# agents/base_agent.py
# Update to use new progress tracking

def _check_run_progress(self, run_info: Dict[str, Any]) -> Dict[str, Any]:
    """Check progress of a single run using new progress tracker."""
    work_dir = run_info['work_dir']
    config = run_info.get('config', {})
    
    # Use new progress tracker
    tracker = create_progress_tracker(work_dir, config)
    progress = tracker.get_progress()
    
    # Convert to agent-friendly format
    return {
        'run_name': run_info['run_name'],
        'runner_type': progress.runner_type,
        'status': progress.status,
        'progress': progress.progress_percentage,
        'details': {
            'completed_epochs': progress.completed_epochs,
            'total_epochs': progress.total_epochs,
            'early_stopped': progress.early_stopped,
            'scores': progress.latest_score,
        }
    }
```

## Implementation Steps

1. **Phase 1: Core Infrastructure**
   - Implement BaseProgressTracker abstract class
   - Implement TrainerProgressTracker (refactor existing logic)
   - Implement EvaluatorProgressTracker (new functionality)
   - Create progress tracker factory with runner type detection

2. **Phase 2: Integration**
   - Update run_status.py to use progress trackers
   - Update session_progress.py for unified progress reporting
   - Update base_agent.py to handle both runner types
   - Update agent_log_parser.py for evaluator log patterns

3. **Phase 3: Enhanced Features**
   - Add score tracking integration (like eval_viewer)
   - Implement caching with invalidation
   - Add support for multi-stage training detection
   - Create unified progress viewer UI

4. **Phase 4: Testing & Documentation**
   - Write unit tests for each progress tracker
   - Test backward compatibility with existing trainer runs
   - Update documentation with new API
   - Add examples for both trainer and evaluator usage

## Backward Compatibility

1. **Existing API preserved**: Functions like `get_run_status()` maintain same interface
2. **Default behavior**: Assumes trainer if runner type cannot be detected
3. **Progress.json format**: Extended with new fields but old fields preserved
4. **Config compatibility**: Works with existing configs without modification

## Benefits

1. **Unified System**: Single API for all progress tracking needs
2. **Better Evaluator Support**: First-class support for evaluation runs
3. **Score Tracking**: Integrated best/latest score tracking
4. **Extensibility**: Easy to add new runner types
5. **Performance**: Caching reduces repeated file I/O
6. **Consistency**: Aligns with eval_viewer patterns

## Future Extensions

1. **Multi-Stage Training**: Support for complex training pipelines
2. **Distributed Training**: Track progress across multiple nodes
3. **Real-time Updates**: WebSocket-based live progress monitoring
4. **Progress Predictions**: ML-based ETA predictions
5. **Resource Tracking**: GPU/CPU usage integration