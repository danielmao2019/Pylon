# Race Condition Error Explanation

## The Error

```
RuntimeError: Failed to load progress file ./logs/benchmarks/change_detection/levir_cd/ChangeMamba-Base_run_2/progress.json: Expecting value: line 1 column 1 (char 0)
```

This error occurred during concurrent execution of the launcher with `num_jobs=2`, where multiple threads were processing different experiment configurations simultaneously.

## Root Cause Analysis

### 1. **Concurrent Thread Execution**
- The launcher uses `ThreadPoolExecutor` to process multiple config files in parallel
- Each thread calls `get_run_status()` → `tracker.get_progress()` → `get_session_progress()`
- Multiple threads can potentially access the same `progress.json` file simultaneously

### 2. **Time-of-Check to Time-of-Use (TOCTOU) Race Condition**

The original code had this vulnerable pattern:

```python
# Thread A and B both execute this simultaneously
if os.path.exists(progress_file):  # Both check: False (file doesn't exist)
    # ... try to read file
else:
    # Both threads enter here
    return _compute_and_cache_progress(work_dir, expected_files)
```

### 3. **The Race Condition Scenarios**

**Scenario 1: Write-Write Race**
1. Thread A checks `os.path.exists(progress_file)` → `False`
2. Thread B checks `os.path.exists(progress_file)` → `False`  
3. Both threads call `_compute_and_cache_progress()`
4. Both threads try to write to the same `progress.json` file
5. Thread A opens file and starts writing
6. Thread B opens file and overwrites/truncates Thread A's data
7. Result: Corrupted/empty file

**Scenario 2: Read-Write Race**
1. Thread A starts writing `progress.json`
2. Thread B checks `os.path.exists(progress_file)` → `True` (file was just created)
3. Thread B checks `os.path.getsize(progress_file)` → `> 0` (file has some data)
4. Thread B tries to read JSON while Thread A is still writing
5. Thread B reads incomplete JSON (e.g., `{"completed_ep`) 
6. `json.load()` fails with "Expecting value: line 1 column 1 (char 0)"

### 4. **Why the Error Message is Misleading**

The error "Expecting value: line 1 column 1 (char 0)" suggests an empty file, but the file actually contained valid JSON when checked individually. This happens because:

- The JSON parser received incomplete/corrupted data during the race condition
- By the time we manually inspected the file, it had been properly written
- The race condition is timing-dependent and intermittent

## Technical Details

### **File I/O Without Atomic Operations**

Original `save_json()` implementation:
```python
with open(filepath, mode='w') as f:
    f.write(json_content)  # NOT atomic - other processes can read partial data
```

### **No Concurrency Control**

Original `get_session_progress()` had no locking:
```python
with open(progress_file, 'r') as f:
    data = json.load(f)  # Can read while another thread is writing
```

## The Fix Applied

### 1. **Per-File Threading Locks** (`session_progress.py`)
```python
# Global lock registry for per-file synchronization
_progress_file_locks = {}
_progress_locks_lock = threading.Lock()

def _get_progress_lock(progress_file: str) -> threading.Lock:
    with _progress_locks_lock:
        if progress_file not in _progress_file_locks:
            _progress_file_locks[progress_file] = threading.Lock()
        return _progress_file_locks[progress_file]

# Thread-safe reading
with _get_progress_lock(progress_file):
    with open(progress_file, 'r') as f:
        data = json.load(f)
```

### 2. **Thread-Safe Computation**
```python
def _compute_and_cache_progress(work_dir: str, expected_files: List[str]) -> ProgressInfo:
    progress_file = os.path.join(work_dir, "progress.json")
    file_lock = _get_progress_lock(progress_file)
    
    with file_lock:  # Only one thread can compute per file
        # Double-check pattern after acquiring lock
        result = _safe_read_progress_file(progress_file)
        if result is not None:
            return result
        # ... compute and save
```

### 3. **Code Reuse Pattern**
```python
def _safe_read_progress_file(progress_file: str) -> Optional[ProgressInfo]:
    """Reusable thread-safe reading logic."""
    # Centralized read logic used in multiple places
```

## Why This Solution Works

1. **Consistent with Codebase**: Uses `threading.Lock()` pattern already established throughout Pylon
2. **Thread Synchronization**: Per-file locks ensure only one thread accesses each progress file at a time
3. **Double-Check Pattern**: Prevents redundant computation when multiple threads race to the same file
4. **Code Reuse**: Centralized `_safe_read_progress_file()` eliminates duplicate read logic
5. **Memory Efficient**: Lock registry only creates locks for files that are actually accessed

## Why Threading Locks vs File Locks

**Initial approach used `fcntl` (file locking):**
- ❌ First time `fcntl` used in this codebase (inconsistent)
- ❌ Overkill for same-process thread synchronization  
- ❌ More complex error handling (lock acquisition failures)

**Final approach uses `threading.Lock()`:**
- ✅ Consistent with existing patterns (30+ uses in codebase)
- ✅ Simpler and more appropriate for thread-level race conditions
- ✅ Better performance (no system calls)
- ✅ Easier testing and debugging

## Key Lesson

This demonstrates why **defensive programming** (adding error handling) would have been the wrong approach. The instinct might be:

```python
# ❌ WRONG - Masks the real bug
try:
    data = json.load(f)
except json.JSONDecodeError:
    return fallback_data()  # Hides race condition!
```

Instead, **investigating the root cause** revealed a fundamental concurrency issue that needed proper synchronization primitives (file locking and atomic operations).