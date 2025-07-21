# Generalized Progress Tracking Implementation Plan

## Overview

This document outlines the plan to generalize Pylon's progress tracking functionality to support both BaseTrainer and BaseEvaluator results. Currently, the progress tracking is tailored specifically for trainer results with epoch-based directory structures. This plan will create a unified system that can handle different runner types while maintaining backward compatibility.

## Current State Analysis

### Current API Structure
- **RunStatus**: Contains `config`, `work_dir`, `progress` (ProgressInfo), `status`, `process_info`
- **ProgressInfo**: Contains `completed_epochs`, `progress_percentage`, `early_stopped`, `early_stopped_at_epoch`
- **get_all_run_status()**: Returns `Dict[str, RunStatus]` (NOT strings)
- **get_run_status()**: Returns `RunStatus` object (NOT string)

### Trainer Progress Tracking
- **Structure**: Epoch-based directories (`epoch_0/`, `epoch_1/`, etc.)
- **Expected Files**: `["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]`
- **Progress Calculation**: `completed_epochs / total_epochs * 100`
- **Log Pattern**: `train_val*.log`
- **Status**: running, finished, stuck, failed, outdated
- **ProgressInfo**: Full dataclass with epoch information

### Evaluator Progress Tracking (Current Gaps)
- **Structure**: Flat directory (no epochs)
- **Expected Files**: `["evaluation_scores.json"]` (from BaseEvaluator.expected_files)
- **Progress**: Binary (0% or 100%) but status has same options as trainer
- **Log Pattern**: `eval_*.log` 
- **Status**: running, finished, stuck, failed, outdated (same as trainer)
- **No epoch-based progress tracking currently implemented**

### Eval Viewer Patterns (Already Handles Both)
- **Runner Type Detection**: Checks for specific file patterns  
- **Shape Handling**: `(N, C, H, W)` for trainer, `(C, H, W)` for evaluator
- **Unified Display**: Single interface for both result types

## Design Goals

1. **Fail Fast**: Enforce runner type detection - no fallbacks or guessing
2. **ProgressInfo Dataclass**: Always return ProgressInfo dataclass, not percentages
3. **Unified Status Options**: Same status options for both trainers and evaluators
4. **Backward Compatibility**: Existing API signatures preserved
5. **Separate Module**: New progress tracking at `@utils/automation/progress_tracking/`
6. **Multi-stage Ready**: Design supports but doesn't implement multi-stage training
7. **WebSocket Ready**: Architecture supports real-time updates

## Architecture Design

### 1. New Module Structure

```
utils/automation/progress_tracking/
├── __init__.py
├── base_progress_tracker.py      # Abstract base class
├── trainer_progress_tracker.py   # Trainer-specific implementation  
├── evaluator_progress_tracker.py # Evaluator-specific implementation
├── progress_tracker_factory.py   # Factory and runner detection
└── websocket_progress.py         # Future: Real-time WebSocket updates
```

### 2. Enhanced ProgressInfo Dataclass

```python
# utils/automation/progress_tracking/base_progress_tracker.py
from typing import Dict, Any, Optional, Literal, List
from dataclasses import dataclass
import time

@dataclass  
class ProgressInfo:
    """Enhanced progress information structure for both trainers and evaluators."""
    # REQUIRED: Existing fields (backward compatibility)
    completed_epochs: int
    progress_percentage: float
    early_stopped: bool = False
    early_stopped_at_epoch: Optional[int] = None
    
    # NEW: Runner type identification
    runner_type: Literal['trainer', 'evaluator', 'multi_stage'] = 'trainer'
    
    # NEW: Enhanced metadata
    total_epochs: Optional[int] = None  # None for evaluators
    start_time: Optional[float] = None
    last_update_time: Optional[float] = None
    
    # NEW: Score tracking (like eval_viewer)
    best_score: Optional[Dict[str, float]] = None
    latest_score: Optional[Dict[str, float]] = None
    
    # NEW: Error handling
    error_message: Optional[str] = None
    warnings: Optional[List[str]] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization (backward compatibility)."""
        return asdict(self)
```

### 3. Abstract Base Class

```python
# utils/automation/progress_tracking/base_progress_tracker.py
from abc import ABC, abstractmethod

class BaseProgressTracker(ABC):
    """Abstract base class for progress tracking."""
    
    def __init__(self, work_dir: str, config: Optional[Dict[str, Any]] = None):
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

### 4. Trainer Progress Tracker

```python
# utils/automation/progress_tracking/trainer_progress_tracker.py
class TrainerProgressTracker(BaseProgressTracker):
    """Progress tracker for BaseTrainer runs."""
    
    def get_runner_type(self) -> Literal['trainer']:
        return 'trainer'
    
    def get_expected_files(self) -> List[str]:
        return ["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]
    
    def get_log_pattern(self) -> str:
        return "train_val*.log"
    
    def calculate_progress(self) -> ProgressInfo:
        """Calculate trainer-specific progress using existing session_progress logic."""
        # Use existing logic from session_progress.py
        from utils.automation.run_status.session_progress import get_session_progress
        
        # Get basic progress info (preserves existing logic)  
        basic_progress = get_session_progress(self.work_dir, self.get_expected_files())
        
        # Enhance with new fields
        total_epochs = self.config.get('epochs') if self.config else None
        
        # Load scores if available (like eval_viewer does)
        best_score, latest_score = self._load_validation_scores()
        
        return ProgressInfo(
            # Existing fields (backward compatibility)
            completed_epochs=basic_progress.completed_epochs,
            progress_percentage=basic_progress.progress_percentage,
            early_stopped=basic_progress.early_stopped,
            early_stopped_at_epoch=basic_progress.early_stopped_at_epoch,
            
            # New fields
            runner_type='trainer',
            total_epochs=total_epochs,
            best_score=best_score,
            latest_score=latest_score,
        )
    
    def is_complete(self) -> bool:
        """Check if trainer is complete."""
        if not self.config or 'epochs' not in self.config:
            return False
        return self.get_progress().completed_epochs >= self.config['epochs']
```

### 5. Evaluator Progress Tracker

```python
# utils/automation/progress_tracking/evaluator_progress_tracker.py
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
        eval_complete = self.is_complete()
        
        # Binary progress: 0% or 100%
        progress_percentage = 100.0 if eval_complete else 0.0
        
        # For evaluators: completed_epochs = 1 if finished, 0 if not
        completed_epochs = 1 if eval_complete else 0
        
        # Load evaluation scores if available
        scores = self._load_evaluation_scores() if eval_complete else None
        
        return ProgressInfo(
            # Existing fields (adapted for evaluator)
            completed_epochs=completed_epochs,
            progress_percentage=progress_percentage,
            early_stopped=False,  # Not applicable for evaluators
            early_stopped_at_epoch=None,
            
            # New fields
            runner_type='evaluator',
            total_epochs=1,  # Evaluators have conceptually 1 "epoch"
            best_score=scores,  # Same as latest for single evaluation
            latest_score=scores,
        )
    
    def is_complete(self) -> bool:
        """Check if evaluator is complete."""
        return all(
            os.path.exists(os.path.join(self.work_dir, f)) and
            os.path.getsize(os.path.join(self.work_dir, f)) > 0
            for f in self.get_expected_files()
        )
```

### 6. Progress Tracker Factory with Fail Fast

```python
# utils/automation/progress_tracking/progress_tracker_factory.py
def detect_runner_type(work_dir: str, config: Optional[Dict[str, Any]] = None) -> Literal['trainer', 'evaluator']:
    """Detect runner type from work_dir structure or config. FAIL FAST if cannot determine."""
    
    # Strategy 1: Check existing files (like eval_viewer does)
    if os.path.exists(os.path.join(work_dir, "evaluation_scores.json")):
        return 'evaluator'
    
    if os.path.exists(os.path.join(work_dir, "epoch_0")):
        return 'trainer'
    
    # Strategy 2: Check config if available
    if config:
        runner_class = config.get('runner', {}).get('class', None)
        if runner_class:
            class_name = runner_class.__name__ if hasattr(runner_class, '__name__') else str(runner_class)
            if 'Evaluator' in class_name:
                return 'evaluator' 
            elif 'Trainer' in class_name:
                return 'trainer'
                
        # Strategy 3: Check for 'epochs' field (trainers have this)
        if 'epochs' in config:
            return 'trainer'
    
    # FAIL FAST: Cannot determine runner type
    available_files = os.listdir(work_dir) if os.path.exists(work_dir) else []
    config_info = f"Config keys: {list(config.keys())}" if config else "No config provided"
    
    raise ValueError(
        f"Cannot detect runner type for work_dir: {work_dir}\n"
        f"Available files: {available_files}\n"
        f"{config_info}\n"
        f"Expected patterns:\n"
        f"  - Trainer: epoch_0/ directory OR 'epochs' in config\n"
        f"  - Evaluator: evaluation_scores.json file OR 'Evaluator' in runner class name"
    )

def create_progress_tracker(work_dir: str, config: Optional[Dict[str, Any]] = None) -> BaseProgressTracker:
    """Factory function to create appropriate progress tracker."""
    runner_type = detect_runner_type(work_dir, config)
    
    if runner_type == 'evaluator':
        return EvaluatorProgressTracker(work_dir, config)
    else:
        return TrainerProgressTracker(work_dir, config)
```

### 7. Updated Run Status Integration

```python
# utils/automation/run_status/run_status.py
# Import from new progress tracking module
from utils.automation.progress_tracking import create_progress_tracker

def get_run_status(
    config: str,
    expected_files: List[str],  # Keep for backward compatibility but not used  
    epochs: int,               # Keep for backward compatibility but not used
    config_to_process_info: Dict[str, ProcessInfo],
    sleep_time: int = 86400,
    outdated_days: int = 30
) -> RunStatus:
    """UPDATED: Get run status using new progress tracker."""
    
    work_dir = get_work_dir(config)
    config_dict = load_config(config)  # Load actual config
    
    # Create progress tracker (handles both trainer and evaluator)
    tracker = create_progress_tracker(work_dir, config_dict)
    progress = tracker.get_progress()
    
    # Determine status using existing logic but with tracker data
    log_last_update = get_log_last_update(work_dir, tracker.get_log_pattern())
    epoch_last_update = get_epoch_last_update(work_dir, tracker.get_expected_files())
    
    is_running_status = log_last_update is not None and (time.time() - log_last_update <= sleep_time)
    
    if is_running_status:
        status: _RunStatus = 'running'
    elif tracker.is_complete():
        if epoch_last_update is not None and (time.time() - epoch_last_update > outdated_days * 24 * 60 * 60):
            status = 'outdated'
        else:
            status = 'finished'
    elif config in config_to_process_info:
        status = 'stuck'
    else:
        status = 'failed'
    
    process_info = config_to_process_info.get(config, None)
    
    return RunStatus(
        config=config,
        work_dir=work_dir,
        progress=progress,  # Now includes runner_type and enhanced info
        status=status,
        process_info=process_info
    )

def get_log_last_update(work_dir: str, log_pattern: str) -> Optional[float]:
    """Updated to use dynamic log pattern."""
    if not os.path.isdir(work_dir):
        return None
    logs = glob.glob(os.path.join(work_dir, log_pattern))
    if len(logs) == 0:
        return None
    return max([os.path.getmtime(fp) for fp in logs])
```

## Implementation Steps

1. **Phase 1: Core Infrastructure**
   - Create new `utils/automation/progress_tracking/` module
   - Implement enhanced ProgressInfo dataclass with new fields
   - Implement BaseProgressTracker abstract class
   - Implement TrainerProgressTracker (wraps existing session_progress logic)
   - Implement EvaluatorProgressTracker (new functionality)
   - Create progress tracker factory with fail-fast runner type detection

2. **Phase 2: Integration**
   - Update run_status.py to use progress trackers
   - Keep session_progress.py as-is but import from new module
   - Update agents to work with RunStatus.progress (ProgressInfo object)
   - Remove any remaining string-based status handling

3. **Phase 3: Enhanced Features**  
   - Add score tracking integration (like eval_viewer)
   - Implement progress.json caching with new fields
   - Add WebSocket-based real-time progress updates
   - Create unified progress viewer UI

4. **Phase 4: Testing & Documentation**
   - Write unit tests for each progress tracker
   - Test fail-fast runner detection
   - Test backward compatibility with existing trainer runs
   - Add examples for both trainer and evaluator usage

## Progress.json Fields

### Old Fields (Preserved)
```json
{
  "completed_epochs": 5,
  "progress_percentage": 50.0,  
  "early_stopped": false,
  "early_stopped_at_epoch": null
}
```

### New Fields (Added)
```json
{
  // Existing fields preserved above...
  "runner_type": "trainer",
  "total_epochs": 10,
  "start_time": 1642680000.0,
  "last_update_time": 1642680300.0,
  "best_score": {"accuracy": 0.95, "loss": 0.05},
  "latest_score": {"accuracy": 0.93, "loss": 0.07},
  "error_message": null,
  "warnings": []
}
```

**Backward Compatibility**: Old progress.json files still work, new fields added only when using new progress trackers.

## Agent Integration 

**Current Issue**: The agents assume `RunStatus.progress` is a ProgressInfo object but some code might treat status as string.

**Solution**: Agents already receive `RunStatus` objects with `.progress` field containing ProgressInfo. No changes needed to agent logic, just ensure all status handling uses the object properly:

```python
# In agents - this already works correctly:
all_run_status = get_all_run_status(...)  # Returns Dict[str, RunStatus]
for run_name, run_status in all_run_status.items():
    progress_info = run_status.progress  # ProgressInfo object
    progress_percentage = progress_info.progress_percentage
    runner_type = progress_info.runner_type  # NEW field
```

## WebSocket Real-time Updates

**Benefits of WebSocket Implementation**:

1. **Live Progress Monitoring**: Real-time updates without polling
2. **Multi-client Support**: Multiple users can monitor same experiments
3. **Efficient Bandwidth**: Only send changes, not full state
4. **Event-driven Architecture**: React to file system changes immediately
5. **Better User Experience**: No page refresh needed for updates

**WebSocket Architecture**:
```python
# utils/automation/progress_tracking/websocket_progress.py
class ProgressWebSocketServer:
    """Real-time progress updates via WebSocket."""
    
    def __init__(self, work_dirs: List[str]):
        self.progress_trackers = {
            work_dir: create_progress_tracker(work_dir, load_config(work_dir))
            for work_dir in work_dirs
        }
        self.clients = set()
        
    async def monitor_progress(self):
        """Background task to monitor file changes and broadcast updates."""
        while True:
            for work_dir, tracker in self.progress_trackers.items():
                new_progress = tracker.get_progress(force_refresh=True)
                if self._has_changes(work_dir, new_progress):
                    await self._broadcast_update(work_dir, new_progress)
            await asyncio.sleep(1)  # Check every second
```

**Use Cases**:
- **Experiment Dashboard**: Live overview of all running experiments
- **Debug Monitoring**: Watch training metrics in real-time
- **Resource Management**: Immediate notification when GPUs become available
- **Collaboration**: Share experiment progress with team members

## Fail Fast Runner Detection

**Philosophy**: Better to fail immediately with clear error than guess wrong and cause subtle bugs.

**Detection Strategies (in order)**:
1. **File Pattern Detection**: Check for `evaluation_scores.json` (evaluator) or `epoch_0/` (trainer)
2. **Config Analysis**: Check `runner.class` field for class name patterns  
3. **Field Heuristics**: Check for `epochs` field (indicates trainer)
4. **FAIL**: Raise detailed ValueError with diagnostic information

**Error Message Example**:
```
ValueError: Cannot detect runner type for work_dir: /path/to/logs/experiment_1
Available files: ['config.json', 'git_0.log']
Config keys: ['model', 'dataset', 'runner']
Expected patterns:
  - Trainer: epoch_0/ directory OR 'epochs' in config
  - Evaluator: evaluation_scores.json file OR 'Evaluator' in runner class name
```

## Backward Compatibility Guarantees

1. **API Signatures**: `get_run_status()` and `get_all_run_status()` maintain exact same signatures
2. **Return Types**: Still return `RunStatus` and `Dict[str, RunStatus]` respectively  
3. **ProgressInfo Fields**: All existing fields (`completed_epochs`, `progress_percentage`, etc.) preserved
4. **Existing progress.json**: Old files still load correctly, new fields added transparently
5. **Agent Integration**: Existing agent code continues to work without modification

## Benefits Summary

1. **Unified System**: Same API for trainers, evaluators, and future runner types
2. **First-class Evaluator Support**: Proper progress tracking for evaluation jobs
3. **Enhanced Metadata**: Score tracking, timing info, error reporting
4. **Real-time Updates**: WebSocket support for live monitoring
5. **Fail Fast Detection**: Clear errors instead of silent wrong assumptions
6. **Performance**: Smart caching with file-based invalidation  
7. **Extensibility**: Easy to add multi-stage training, distributed training, etc.
8. **Consistency**: Aligns with eval_viewer's proven patterns