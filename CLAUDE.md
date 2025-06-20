# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Pylon is a PyTorch-based deep learning framework for computer vision research, supporting both 2D vision tasks (change detection, segmentation, object detection) and 3D vision tasks (point cloud registration, 3D change detection) with extensive multi-task learning capabilities.

## Essential Commands

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/models/change_detection/test_change_star.py

# Run with verbose output
pytest -v --tb=short tests/
```

### Training
```bash
# Basic training
python main.py --config-filepath configs/examples/linear/config.py

# Debug mode (3 epochs, small batches)
python main.py --config-filepath configs/examples/linear/config.py --debug
```

### Dataset Viewer
```bash
# Launch web-based dataset viewer
python -m data.viewer.cli

# Custom host/port
python -m data.viewer.cli --host 0.0.0.0 --port 8050
```

### Code Quality
```bash
# Format code
black .
isort .

# Lint
flake8 .
```

## Architecture

### Core Design Philosophy
Pylon follows several fundamental design patterns that enable extensible, reproducible, and high-performance computer vision research:

### 1. Dataset Design Pattern
**Every dataset follows the three-field structure:**
- `inputs`: Dictionary with actual input data (images, point clouds, etc.)
- `labels`: Dictionary with ground truth labels for supervision
- `meta_info`: Dictionary with metadata (paths, sizes, indices, etc.)

**Implementation workflow:**
1. Define `self.annotations` in `_init_annotations()` - list of annotation dictionaries
2. Implement `_load_datapoint()` using `self.annotations` and helper methods
3. Use helper methods like `_load_seg_labels()`, `_load_amodal_masks()` for modular loading
4. `meta_info` = `self.annotations` + lightweight additional information

### 2. Asynchronous Buffer Pattern
**Critical for GPU utilization:** Criteria, metrics, and optimizers use asynchronous buffers to prevent blocking training loops:
- Background thread: `threading.Thread(target=self._buffer_worker, daemon=True)`
- Thread-safe queue: `queue.Queue()` for non-blocking data collection
- Lock-protected buffer access: `threading.Lock()` for thread safety
- Async tensor operations: Detach from computation graph and move to CPU in background
- `add_to_buffer()` is non-blocking; `summarize()` waits for queue to empty

### 3. Configuration-Driven Architecture
**`build_from_config()` pattern enables flexible component instantiation:**
```python
config = {
    'class': SomeClass,
    'args': {...}
}
obj = build_from_config(config)
```
- Recursive construction of nested objects
- Class references (not strings) for type safety
- Parameter merging and preservation
- Supports PyTorch parameter special handling

### 4. Multi-Task Learning Architecture
**Wrapper pattern for orchestrating multiple tasks:**
- `MultiTaskCriterion` uses `torch.nn.ModuleDict` for task-specific criteria
- `MultiTaskMetric` aggregates results from individual task metrics
- `MTLOptimizer` implements gradient manipulation (PCGrad, MGDA, GradNorm)
- Each task component maintains its own buffer and state

### 5. Training Loop Architecture
**Sophisticated training orchestration:**
- **Deterministic execution**: Per-epoch seeding with separate train/val/test seeds
- **Continuous epoch numbering**: Maintains count across resumptions and multi-stage training
- **Asynchronous I/O**: Threaded checkpoint saving and validation scoring
- **GPU monitoring**: Real-time resource tracking during training
- **Robust resumption**: Automatic detection and resumption from last checkpoint

### 6. Data Pipeline Design
**Memory-efficient and reproducible data loading:**

- **Lazy loading**: Data loading happens in two phases for memory efficiency:
  - *Initialization phase*: Only lightweight metadata stored in `self.annotations` (file paths, indices, etc.)
  - *On-demand phase*: Actual data loading via `_load_datapoint(idx)` only when `dataset[idx]` is accessed
  - Raw data is cached after first load; transforms applied on every access for randomness
  - Example: `self.annotations = [{'image': '/path/img1.png'}]` → actual image loaded only when needed

- **Thread safety**: Comprehensive thread-safe concurrent access via `DatasetCache`:
  - All cache operations protected by `threading.Lock()` for multi-threaded DataLoader workers
  - Deep copying (`copy.deepcopy()`) ensures memory isolation between threads
  - LRU cache updates, hit/miss statistics, and memory management are all thread-safe
  - Lock recreation after pickle/unpickle for multiprocessing support

- **LRU caching**: Memory-aware caching with system monitoring
- **Transform composition**: Functional composition with seed propagation
- **Collation strategy**: Nested dictionary support with custom per-key collators

### Key Directories and Components
- `/configs/`: Template-based experiment configurations with automated generation
- `/data/`: Datasets with caching, transforms, collators, and interactive viewer
- `/criteria/`: Loss functions with asynchronous buffer pattern
- `/metrics/`: Evaluation metrics with threading and buffer management
- `/optimizers/`: Standard and multi-task optimizers with gradient manipulation
- `/runners/`: Training loops with deterministic execution and GPU monitoring
- `/runners/eval_viewer/`: Web-based evaluation result visualization
- `/schedulers/`: Learning rate schedulers with warmup and multi-component support
- `/utils/`: Core utilities including builders, automation, determinism, and monitoring

### Special Utilities
**Automation and Distributed Training:**
- **SSH connection pooling**: Thread-safe multi-server experiment management
- **GPU monitoring**: Real-time utilization tracking across servers
- **Run status monitoring**: Automatic detection of stuck/failed/finished experiments
- **Config generation**: Automated creation of experiment variants with different seeds

**Determinism and Reproducibility:**
- **Comprehensive seeding**: Per-epoch, per-phase random seed management
- **State preservation**: Robust checkpoint and resumption handling
- **Validation**: Extensive configuration and type checking

### Testing Philosophy and Patterns
**Comprehensive testing approach with standardized patterns:**

#### **Common Test Pattern Taxonomy**

1. **Correctness Verification Pattern**
   - Hard-coded inputs with known expected outputs
   - Mathematical validation against analytical solutions
   - Example: `test_confusion_matrix.py` with specific tensor inputs and expected matrices

2. **Equivalence Testing Pattern**
   - Compare re-implementations against reference/official implementations
   - Tolerance-based equality using `torch.allclose()` for numerical comparisons
   - Example: `test_pcr_collator.py` comparing new vs ground truth implementations

3. **Random Ground Truth Pattern**
   - Generate controlled random data where ground truth is known
   - Apply known transformations (rotations, translations) and verify results
   - Seeded random generation for reproducible test cases

4. **Edge Case Testing Pattern**
   - Boundary conditions: empty inputs, single elements, extreme values
   - Special cases: NaN, inf, zero-length tensors, minimal valid inputs
   - Example: `test_chamfer_distance.py` testing empty point clouds

5. **Invalid Input Testing Pattern**
   - Type mismatches, incompatible shapes, invalid value ranges
   - Exception verification using `pytest.raises()` with specific error types
   - Input validation and error message testing

6. **Initialization Testing Pattern**
   - Verify proper object setup and internal state consistency
   - Attribute existence, module registration (for PyTorch components)
   - State validation after initialization

7. **Determinism Testing Pattern**
   - Same seed produces identical results across multiple runs
   - Cross-platform consistency and reproducible behavior
   - Transform and random operation reproducibility

8. **Resource Testing Pattern**
   - Memory usage monitoring and leak detection
   - GPU memory management and cleanup verification
   - File handle and resource cleanup testing
   - Performance characteristics and resource bounds
   - Example: `test_geotransformer.py` monitoring memory usage within specific bounds

9. **Concurrency Testing Pattern**
   - Thread safety with multiple concurrent workers
   - Race condition detection and prevention
   - Lock behavior and deadlock prevention
   - Multi-process data sharing and synchronization
   - Example: `test_dataset_cache.py` testing concurrent cache access

#### **Test Quality Assessment**

**Well-Implemented Examples:**
- `test_confusion_matrix.py`: Comprehensive parametrized tests with edge cases
- `test_focal_loss.py`: Thorough input validation and reference comparison
- `test_point_cloud_ops.py`: Equivalence testing with comprehensive coverage
- `test_geotransformer.py`: Memory usage monitoring and performance testing

**Files Needing Improvement:**
- Dataset tests (`test_air_change_dataset.py`, etc.): Only basic iteration, missing edge cases
- Model tests (`test_dsam_net.py`, etc.): Only forward pass, missing gradient/validation testing
- Transform tests: Limited determinism and edge case coverage

#### **Recommended Test Templates**

**For Dataset Classes:**
```python
def test_initialization()        # Initialization pattern
def test_basic_functionality()   # Hard-coded correctness
def test_transforms()           # Determinism pattern  
def test_edge_cases()           # Edge case pattern
def test_invalid_inputs()       # Invalid input pattern
```

**For Model Classes:**
```python
def test_forward_pass()         # Basic correctness
def test_gradient_flow()        # Gradient verification
def test_input_validation()     # Invalid input pattern
def test_memory_usage()         # Resource management
```

**For Loss/Metric Functions:**
```python
def test_known_cases()          # Hard-coded correctness
def test_reference_implementation() # Equivalence pattern
def test_edge_cases()           # Edge case pattern
def test_mathematical_properties()  # Random ground truth
```

### C++ Extensions
Some modules require building:
```bash
# GeoTransformer
cd data/collators/geotransformer && python setup.py install && cd ../../..

# Buffer/OverlapPredator  
cd data/collators/buffer/cpp_wrappers && bash compile_wrappers.sh && cd ../../../..
```

### Error Handling Philosophy
**Avoid unnecessary try-except blocks - only use when truly necessary:**

- **DO NOT add try-except blocks by default** - they hide error sources and make debugging inefficient
- **Let errors propagate naturally** - Python's stack traces show exact error locations
- **Only use try-except when necessary** for specific functionality, not for general "robustness"

**Examples of necessary try-except usage:**
```python
# Example 1: _call_with_seed method - API compatibility for seed parameter
def _call_with_seed(func, op_inputs, seed=None):
    try:
        if len(op_inputs) == 1:
            op_outputs = [func(op_inputs[0], seed=seed)]
        else:
            op_outputs = func(*op_inputs, seed=seed)
    except Exception as e:
        if "got an unexpected keyword argument 'seed'" in str(e):
            if len(op_inputs) == 1:
                op_outputs = [func(op_inputs[0])]
            else:
                op_outputs = func(*op_inputs)
        else:
            raise

# Example 2: _call_single_with_generator method - API compatibility for generator parameter
def _call_single_with_generator(self, *args, generator):
    try:
        return self._call_single(*args, generator=generator)
    except Exception as e:
        if "got an unexpected keyword argument 'generator'" in str(e):
            return self._call_single(*args)
        else:
            raise
```

**Key principles:**
- Use assertions for input validation instead of try-except
- Prefer explicit checks over catching exceptions

### Code Style Guidelines

#### Import Statement Order
**Always follow this exact order with NO spaces between imports:**
```python
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import sys
import copy
import numpy as np
import torch
import torchvision
from data.datasets.base_dataset import BaseDataset
from criteria.focal_loss import FocalLoss
from utils.builders.builder import build_from_config
```

1. `from typing import` - always first
2. Python native packages (`os`, `sys`, `copy`, etc.)
3. External packages (`numpy`, `torch`, `torchvision`, etc.)
4. Project modules using **full file paths** (not module imports)

#### Config vs Source Code Import Patterns
- **Source code**: Use full file paths - `from data.datasets.base_dataset import BaseDataset`
- **Config files**: Use module imports - `from data.datasets import BaseDataset` (user-friendly)
- **Note**: Config files are program-generated, so manual editing is rare

#### Type Annotations
**Always include type annotations for function/method arguments and return types:**
```python
def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
    # implementation

def build_from_config(config: Dict[str, Any], **kwargs: Any) -> Any:
    # implementation
```

#### Documentation Strings
**Not all functions need docstrings, but when used, follow this pattern:**
```python
def some_function(arg1: int, arg2: str) -> bool:
    """Brief description of what the function does.
    
    Args:
        arg1: Description of first argument
        arg2: Description of second argument
        
    Returns:
        Description of return value
    """
```

#### Function and File Organization
**Break down complex code for maintainability:**
- **Long functions**: Break down using helper functions with clear single responsibilities
- **Long files**: Split into multiple files and organize as modules (folders with `__init__.py`)
- **Test organization**: Group tests by patterns/philosophies rather than single large files

#### Testing Implementation Guidelines
**Use pytest with best practices:**
- **Framework**: Use `pytest` (not unittest test classes)
- **Parametrization**: Use `@pytest.mark.parametrize` for multiple test cases
- **Test organization**: For large test modules, split by test patterns into separate files:
  ```
  tests/criteria/base_criterion/
  ├── __init__.py
  ├── test_initialization.py      # Initialization pattern
  ├── test_buffer_management.py   # Threading/async buffer tests
  ├── test_device_handling.py     # Device transfer tests
  ├── test_edge_cases.py          # Error handling and edge cases
  └── test_determinism.py         # Reproducibility tests
  ```

#### Testing Focus by Code Origin
**Different testing approaches based on code source:**
- **Your implementation** (base classes, wrappers): Comprehensive testing with all 9 patterns
- **Copied from official repos** (domain-specific models/losses): Integration testing only
  - Test successful execution with dummy inputs
  - Verify basic input/output shapes and types
  - No need to verify mathematical correctness

### Important Implementation Notes
- Uses PyTorch 2.0.0 with CUDA 11.8
- Follows OpenMMLab conventions (mmengine, mmcv, mmdet)
- Emphasizes Python-native objects and inheritance for extensibility
- Dictionary-as-tensor operations for flexible nested data structures
- Async logging with thread-safe buffered writes