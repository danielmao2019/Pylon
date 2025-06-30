# Launcher CPU Parsing Bug Fix

**Date:** 2025-06-30  
**Priority:** Critical  
**Status:** Fixed  
**Files Affected:**
- `utils/monitor/cpu_status.py`
- `agents/launcher.py`

## Problem Summary

The Pylon launcher was crashing with a `TypeError` when trying to start experiment jobs. The error occurred when comparing CPU utilization statistics with numeric thresholds:

```
TypeError: '<' not supported between instances of 'NoneType' and 'int'
```

This prevented the launcher from successfully starting any experiments across the distributed computing infrastructure.

## Error Details

**Error Location:** `agents/launcher.py:142`
```python
cpu_util_ok = cpu['cpu_stats']['avg'] < 80
TypeError: '<' not supported between instances of 'NoneType' and 'int'
```

**Stack Trace Context:**
- `launcher.spawn(num_jobs=2)` → 
- `_launch_missing()` → 
- `_find_idle_gpus()` → 
- CPU threshold comparison failure

## Root Cause Analysis

### 1. **Data Flow Investigation**

The error originated from a **string parsing bug** in the CPU monitoring system, not from server connectivity issues. Here's the complete data flow:

```
top command → CPU parsing → monitoring buffers → statistics calculation → launcher comparison
     ↓              ↓              ↓                    ↓                      ↓
   Works        FAILS HERE      Gets None        avg = None           TypeError
```

### 2. **CPU Parsing Logic Bug**

**Location:** `utils/monitor/cpu_status.py:51-62`

**Original Buggy Code:**
```python
cpu_util = None  # Initialize to None to indicate parsing failure
for line in top_output.splitlines():
    if '%Cpu' in line:
        parts = line.split()
        for i, part in enumerate(parts):
            if 'us,' in part:
                try:
                    cpu_util = float(part.replace('us,', ''))  # BUG HERE
                except ValueError:
                    cpu_util = None
                break
        break
```

**Problem:** The code assumed CPU output format was `"4.8us,"` (number+unit together), but actual `top` output format is:

```bash
%Cpu(s):  4.8 us,  2.0 sy,  0.0 ni, 93.1 id,  0.0 wa,  0.0 hi,  0.2 si,  0.0 st
         ↑     ↑
      number  unit (separate parts)
```

**What happened:**
1. `line.split()` produces: `['%Cpu(s):', '4.8', 'us,', '2.0', 'sy,', ...]`
2. Code finds `'us,'` part but tries `'us,'.replace('us,', '')` → empty string `''`
3. `float('')` raises `ValueError`
4. `cpu_util` remains `None`

### 3. **None Propagation Chain**

1. **CPU Window Population** (`utils/monitor/cpu_monitor.py:108`):
   ```python
   cpu_status['cpu_window'].append(cpu_info['current_cpu'])  # Appends None
   ```

2. **Statistics Calculation** (`utils/monitor/cpu_monitor.py:127-139`):
   ```python
   valid_cpu_values = [v for v in cpu_status['cpu_window'] if v is not None]
   if valid_cpu_values:  # Empty list → False
       # Normal stats calculation
   else:
       cpu_status['cpu_stats'] = {'avg': None, 'min': None, 'max': None}  # Sets avg = None
   ```

3. **Insufficient Guard Check** (`agents/launcher.py:141-142`):
   ```python
   if cpu['cpu_stats'] is not None:  # ✓ Passes (it's a dict)
       cpu_util_ok = cpu['cpu_stats']['avg'] < 80  # ❌ None < 80 → TypeError
   ```

### 4. **Verification Evidence**

**Test Script Results:**
```bash
=== Running top command ===
Top command succeeded
Output: %Cpu(s):  4.8 us,  2.0 sy,  0.0 ni, 93.1 id,  0.0 wa,  0.0 hi,  0.2 si,  0.0 st

=== Parsing with buggy logic ===
Parts: ['%Cpu(s):', '4.8', 'us,', '2.0', 'sy,', ...]
Found 'us,' in part: 'us,'
After replacing 'us,': ''
ValueError: could not convert string to float: ''
Final result: None

=== Parsing with fixed logic ===
Found CPU util: 4.8 (from part 1: '4.8')
Final result: 4.8
```

## Solution Implementation

### 1. **Fix CPU Parsing Logic**

**File:** `utils/monitor/cpu_status.py:57-62`

**Fixed Code:**
```python
for i, part in enumerate(parts):
    if 'us,' in part and i > 0:  # Find 'us,' and ensure there's a previous part
        try:
            cpu_util = float(parts[i-1])  # Get the number from the previous part
        except ValueError:
            cpu_util = None  # Keep as None on parsing failure
        break
```

**Key Changes:**
- Added `and i > 0` check to ensure previous part exists
- Changed from `float(part.replace('us,', ''))` to `float(parts[i-1])`
- Now correctly extracts the number from the part **before** the `'us,'` marker

### 2. **Add Defensive Guards**

**File:** `agents/launcher.py:141-142`

**Fixed Code:**
```python
if (cpu['cpu_stats'] is not None and cpu['memory_stats'] is not None and cpu['cpu_cores'] is not None and
    cpu['cpu_stats']['avg'] is not None and cpu['memory_stats']['avg'] is not None and cpu['load_stats']['avg'] is not None):
    cpu_util_ok = cpu['cpu_stats']['avg'] < 80
    cpu_mem_ok = (cpu['max_memory'] - cpu['memory_stats']['avg']) > 4 * 1024
    cpu_load_ok = cpu['load_stats']['avg'] < cpu['cpu_cores']
    cpu_ok = cpu_util_ok and cpu_mem_ok and cpu_load_ok
```

**Key Changes:**
- Added individual None checks for `cpu['cpu_stats']['avg']`
- Added individual None checks for `cpu['memory_stats']['avg']` 
- Added individual None checks for `cpu['load_stats']['avg']`
- Now prevents any None comparisons with numeric thresholds

## Testing and Verification

### 1. **Before Fix**
```bash
$ python project/launch.py
Traceback (most recent call last):
  File "project/launch.py", line 57, in <module>
    launcher.spawn(num_jobs=2)
  File "agents/launcher.py", line 232, in spawn
    done = self._launch_missing(all_running_status, num_jobs=num_jobs)
  File "agents/launcher.py", line 167, in _launch_missing
    idle_gpus: List[Dict[str, Any]] = self._find_idle_gpus(num_jobs)
  File "agents/launcher.py", line 142, in _find_idle_gpus
    cpu_util_ok = cpu['cpu_stats']['avg'] < 80
TypeError: '<' not supported between instances of 'NoneType' and 'int'
```

### 2. **After Fix**
```bash
$ python project/launch.py
[INFO] 2025-06-30 16:01:03 - ==================================================
[INFO] 2025-06-30 16:01:03 - Collecting all running jobs...
[INFO] 2025-06-30 16:01:03 - Removing stuck jobs...
[INFO] 2025-06-30 16:01:03 - Removing outdated jobs...
[INFO] 2025-06-30 16:01:03 - Launching missing jobs...
[INFO] 2025-06-30 16:01:03 - Executing command on daniel@lilac.uwaterloo.ca (GPU-1): ...
[INFO] 2025-06-30 16:01:04 - Executing command on dani@129.97.201.17 (GPU-5): ...
[INFO] 2025-06-30 16:01:04 - Sleeping for 180 seconds...
```

**✅ Success:** Launcher now successfully:
- Monitors CPU/GPU resources across multiple servers
- Launches experiments on idle resources  
- Enters normal monitoring loop
- No more TypeError exceptions

### 3. **CPU Parsing Verification**
```bash
$ python debug_cpu_parsing_fixed.py
CPU line: '%Cpu(s):  6.4 us,  6.8 sy,  0.0 ni, 86.6 id,  0.0 wa,  0.0 hi,  0.2 si,  0.0 st'
Parts: ['%Cpu(s):', '6.4', 'us,', '6.8', 'sy,', '0.0', 'ni,', '86.6', 'id,', ...]
Found CPU util: 6.4 (from part 1: '6.4')
Final CPU utilization: 6.4
```

## Impact Assessment

### **Before Fix:**
- 🔴 **Launcher completely broken** - could not start any experiments
- 🔴 **Distributed computing infrastructure offline** 
- 🔴 **No experiment job scheduling possible**
- 🔴 **CPU monitoring failing system-wide**

### **After Fix:**
- ✅ **Launcher fully operational** across all servers
- ✅ **Distributed experiment scheduling restored**
- ✅ **CPU monitoring working correctly**
- ✅ **Resource utilization monitoring accurate**

## Lessons Learned

### 1. **String Parsing Assumptions**
- Always verify actual output format vs. assumed format
- Test parsing logic with real system outputs
- Handle edge cases in command output parsing

### 2. **Data Validation**
- Add comprehensive None checks at multiple levels
- Don't rely solely on container-level checks (`dict is not None`)
- Validate individual data values before arithmetic operations

### 3. **Error Propagation**
- Consider how None/error values propagate through data pipelines
- Add validation at consumer endpoints, not just producers
- Implement graceful degradation when partial data is unavailable

### 4. **Testing Strategy**
- Test with actual system outputs, not synthetic data
- Create focused unit tests for parsing logic
- Verify end-to-end functionality after fixes

## Future Improvements

1. **Enhanced Error Handling:**
   - Add logging for CPU parsing failures
   - Implement retry logic for temporary parsing failures
   - Add metrics for monitoring system health

2. **Robust Parsing:**
   - Consider using more robust CPU monitoring tools (e.g., `psutil`)
   - Add fallback parsing strategies for different `top` output formats
   - Validate parsing results against expected ranges

3. **Comprehensive Testing:**
   - Add automated tests for CPU monitoring pipeline
   - Create integration tests for launcher resource detection
   - Add monitoring for launcher health and error rates

---

**Fix Verification:** ✅ Complete  
**Deployment Status:** ✅ Applied to `Pylon`  
**Next Steps:** Monitor launcher performance and add comprehensive test coverage
