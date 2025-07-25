# Race Condition Analysis: Concurrent Access to progress.json Files

## Problem Summary

When running `python project/launch.py` and `python project/dashboard.py` simultaneously, the following error occurs:

```
RuntimeError: Error loading JSON from ./logs/benchmarks/change_detection/cdd/Changer-s50_run_2/progress.json: Expecting value: line 1 column 1 (char 0)
```

This error does **not** occur when running the scripts separately.

## Root Cause: File-Level Race Condition Despite Thread-Safe Locks

The error occurs because **the thread-safe locks in `safe_load_json()` only protect against race conditions within the same process**, but `launch.py` and `dashboard.py` are **two separate Python processes running simultaneously**.

## Detailed Technical Analysis

### 1. Process-Level Concurrency Issue

- `launch.py` and `dashboard.py` are separate Python processes
- Each process has its own memory space and thread locks
- The `_json_file_locks` dictionary in `utils/io/json.py` is per-process, not system-wide
- Threading locks cannot coordinate between different processes

### 2. The Exact Race Condition Sequence

1. **Process A** (e.g., `launch.py`) calls `safe_save_json()` to write to `progress.json`
2. **Process B** (e.g., `dashboard.py`) calls `safe_load_json()` to read from `progress.json`
3. During the write operation, the file temporarily becomes empty or corrupted
4. Process B's read happens at exactly this moment, encountering the empty file
5. **Result**: `Expecting value: line 1 column 1 (char 0)` - JSON parser can't parse empty content

### 3. Why Threading Locks Don't Help

```python
# In utils/io/json.py - these locks are PER-PROCESS only
_json_file_locks = {}  # This is separate for each Python process!
_json_locks_lock = threading.Lock()  # Only works within one process
```

The current locking mechanism only prevents race conditions between threads within the same process, but provides no protection against race conditions between separate processes.

### 4. The Write Operation Vulnerability

- When `safe_save_json()` writes to a file, it uses `open(filepath, mode='w')`
- This **truncates the file to 0 bytes first**, then writes content
- There's a brief moment where the file exists but is empty
- If another process reads during this window, it gets the empty file

### 5. File System Behavior

- File operations are not atomic at the OS level
- `os.path.exists()` returns `True` and `os.path.getsize()` can return `0` during this window
- The `safe_load_json()` function correctly detects the file exists but fails the empty file check

## Reproduction

The race condition was successfully reproduced using a test script that simulates:
- One process writing to `progress.json` (including temporary empty file state)
- Another process reading from `progress.json` concurrently
- Result: Exact same error message as reported

## Why It Works Separately But Fails Together

- **Running separately:** Each process runs alone, no concurrent file access
- **Running simultaneously:** Both processes compete for the same `progress.json` files, creating race conditions during the brief write windows

## Classification

This is a **classic file-level race condition** that requires **inter-process coordination**, not just thread-level coordination. The current thread-safe implementation is correct for single-process scenarios but insufficient for multi-process scenarios.

## Potential Solutions Analysis

| Solution | Pros | Cons | Implementation Complexity | Best For |
|----------|------|------|--------------------------|----------|
| **1. File Locking** (fcntl.flock) | • True inter-process synchronization<br>• Prevents all race conditions<br>• Standard OS-level mechanism<br>• Works with existing code structure | • Platform-dependent (Unix/Linux only)<br>• Can cause deadlocks if not handled properly<br>• Blocking operations may impact performance<br>• Requires careful error handling for lock failures | Medium | Production systems where data integrity is critical |
| **2. Atomic Writes** (temp + rename) | • Atomic at filesystem level<br>• No blocking operations<br>• Cross-platform compatible<br>• Elegant and clean solution<br>• Readers never see partial writes | • Slightly more disk I/O (temp file creation)<br>• Requires cleanup of temp files on failure<br>• Small overhead for each write operation | Low | Most scenarios - good balance of safety and performance |
| **3. Process Coordination** (IPC) | • Most robust for complex scenarios<br>• Can handle advanced coordination needs<br>• Scalable to many processes | • High complexity<br>• Requires major architectural changes<br>• Overkill for simple file access<br>• Platform-specific implementations | High | Complex multi-process applications with extensive coordination needs |
| **4. Retry Logic** (exponential backoff) | • Simple to implement<br>• Minimal changes to existing code<br>• Self-healing behavior<br>• Works with any error type | • Doesn't prevent the race condition, just recovers<br>• Can mask underlying issues<br>• Potential for cascading delays<br>• May still fail under high contention | Low | Quick fixes or when combined with other solutions |
| **5. Separate Cache Files** (per-process) | • Completely eliminates contention<br>• Simple implementation<br>• No synchronization overhead<br>• Each process has its own view | • Data inconsistency between processes<br>• Multiple copies of same data<br>• Synchronization challenges<br>• Not suitable for shared progress tracking | Low | When processes need independent views of data |

## Recommended Solution

**Atomic Writes (Solution #2)** is the recommended approach because:

- ✅ **Elegant**: Readers never see partially written files
- ✅ **Cross-platform**: Works on all operating systems
- ✅ **Low complexity**: Minimal changes to existing code
- ✅ **Performance**: No blocking operations
- ✅ **Reliable**: Filesystem-level atomicity guarantees

### Implementation Strategy

```python
def safe_save_json_atomic(obj: Any, filepath: str) -> None:
    """Atomic JSON saving using temp file + rename."""
    temp_filepath = filepath + '.tmp'
    try:
        # Write to temporary file
        _save_json(obj, temp_filepath)
        # Atomic rename (this is the key!)
        os.rename(temp_filepath, filepath)
    except Exception as e:
        # Cleanup temp file if it exists
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        raise RuntimeError(f"Error saving JSON to {filepath}: {e}") from e
```

The solution should maintain the existing thread-safe behavior while adding inter-process safety through atomic writes.