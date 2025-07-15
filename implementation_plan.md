# Progress Check Enhancement Implementation Plan

## Problem Statement

The current progress check system in `utils/automation/run_status.py` only counts completed epochs but doesn't account for early stopping. This causes:

1. **Incorrect Progress Reporting**: A run that early-stopped at epoch 57/100 shows 57% progress instead of 100%
2. **Agent Relaunching**: The launcher agent incorrectly identifies early-stopped runs as "failed" and relaunches them
3. **Resource Waste**: Completed runs are unnecessarily restarted, wasting compute resources
4. **Agent Viewer Issues**: The agent viewer app shows incorrect overall progress for all experiments

## Current Implementation Analysis

### Progress Check Logic (`utils/automation/run_status.py`)
- `get_session_progress()`: Counts consecutive completed epochs (lines 161-176)
- `get_run_status()`: Considers runs finished only when `progress >= epochs` (line 56)
- No awareness of early stopping mechanism

### Early Stopping Integration
- Implemented in `runners/early_stopping.py`
- Has `was_triggered_at_epoch()` method to detect early stopping
- Expected files per epoch: `["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]`

## Proposed Solution Design

### 1. Progress.json File Strategy
Early stopping class creates and maintains `progress.json` file in run directory (sibling to epoch dirs):

```python
# Structure of progress.json:
{
    "completed_epochs": int,        # actual epochs completed
    "progress_percentage": float,   # true progress (0-100) 
    "early_stopped": bool,          # whether early stopping was triggered
    "early_stopped_at_epoch": Optional[int]  # epoch where early stopping occurred
}
```

### 2. Agent Optimization Strategy
- **Fast Path**: Agent reads `progress.json` if it exists and is valid
- **Slow Path**: Agent re-computes progress only if file missing/corrupted/invalid
- **Auto-Repair**: When re-computing, create/overwrite `progress.json`

### 3. Early Stopping Enhancement
Modify `EarlyStopping.update()` method to:
1. Update internal state (existing logic)
2. Write/update `progress.json` every epoch
3. Include early stopping detection in progress calculation

### 4. Configuration Loading Strategy
Follow `main.py` pattern using `importlib.util`:
```python
import importlib.util
spec = importlib.util.spec_from_file_location("config_file", config_filepath)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
config = module.config
```

### 5. Two-Tier Progress Logic
- **With Early Stopping**: Use full early stopping detection + progress.json
- **Without Early Stopping**: Simple completed epochs / total epochs calculation

## Implementation Requirements

### 1. Early Stopping Class Enhancement
Modify `runners/early_stopping.py`:
- Add `_save_progress_json()` method to write progress file
- Update `update()` method to call `_save_progress_json()` after state updates
- Add progress calculation methods (`_calculate_progress_percentage()`, `_get_early_stop_epoch()`)

### 2. Progress Check Enhancement  
Modify `utils/automation/run_status.py`:
- Update `get_session_progress()` to check for `progress.json` first
- Add fallback logic for when early stopping config is missing
- Use `importlib.util` pattern for config loading (following `main.py`)

### 3. Agent Integration
Update agent viewer to use enhanced progress information:
- Display correct overall progress including early-stopped runs
- Handle progress.json format in viewer backend

### 4. Core Implementation Strategy

#### A. No Early Stopping Configured
- Simple logic: `completed_epochs / tot_epochs * 100`
- No progress.json file needed

#### B. Early Stopping Configured  
- Early stopping class writes progress.json every epoch
- Agent reads progress.json for fast progress checking
- Re-compute only if progress.json missing/invalid

### 5. Metric Building Requirement
- Build metric object first (following `base_trainer.py` pattern)  
- Pass metric to early stopping for DIRECTIONS access
- Use `build_from_config()` for metric instantiation

## Updated API Design

### 1. Enhanced get_session_progress()
Replace current function with enhanced version:
```python
def get_session_progress(work_dir: str, expected_files: List[str]) -> int:
    """
    Enhanced progress calculation with early stopping detection.
    Returns completed epochs, or tot_epochs if early stopped.
    """
    # Try fast path: read progress.json
    progress_file = os.path.join(work_dir, "progress.json")
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
            return progress_data['completed_epochs']
        except (json.JSONDecodeError, KeyError):
            pass  # Fall through to slow path
    
    # Slow path: re-compute and create progress.json
    return _compute_and_cache_progress(work_dir, expected_files)
```

### 2. Early Stopping Enhancement
Add progress.json management to `EarlyStopping` class:
```python
def _save_progress_json(self) -> None:
    """Save current progress to progress.json file."""
    progress_data = {
        "completed_epochs": self.last_read_epoch + 1,
        "progress_percentage": self._calculate_progress_percentage(),
        "early_stopped": self.should_stop_early,
        "early_stopped_at_epoch": self._get_early_stop_epoch() if self.should_stop_early else None
    }
    
    progress_file = os.path.join(self.work_dir, "progress.json")
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)
```

### 3. Configuration Loading Utility
Add new utility in `utils/io/config.py`:
```python
def load_config(config_path: str) -> Dict[str, Any]:
    """Load config using importlib.util pattern from main.py"""
    spec = importlib.util.spec_from_file_location("config_file", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config
```

## Implementation Details

### 1. No Backward Compatibility
- Direct replacement of existing functions
- No feature flags or gradual rollout
- Clean implementation without legacy support

### 2. No Error Handling
- Let errors raise naturally
- No fallback strategies or defensive programming
- Simple and direct implementation

### 3. Performance Optimization
- progress.json provides caching automatically
- Fast path for most agent checks
- Slow path only when necessary

### 4. Early Stopping Detection Flow
1. Check if early stopping configured in config
2. If not configured: use simple epoch counting  
3. If configured: build metric, instantiate early stopping, use full detection

## Implementation Steps

1. **Create Config Loading Utility**
   - Add `utils/io/config.py` with `load_config()` function
   - Import and use in run_status.py

2. **Enhance Early Stopping Class**
   - Add `_save_progress_json()` method
   - Add `_calculate_progress_percentage()` method  
   - Add `_get_early_stop_epoch()` method
   - Modify `update()` to write progress.json after state updates

3. **Update Progress Check Logic**
   - Modify `get_session_progress()` for fast/slow path
   - Add early stopping detection logic using config loading
   - Add `_compute_and_cache_progress()` helper function

4. **Update Agent Integration**
   - Ensure agent viewer uses enhanced progress
   - Handle progress.json format in viewer

5. **Testing**
   - Test with/without early stopping configs
   - Test progress.json caching behavior
   - Test agent viewer integration

---

*Implementation plan updated based on user feedback. Ready for implementation.*