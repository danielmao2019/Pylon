# CLAUDE.md <!-- omit in toc -->

## Table of Contents <!-- omit in toc -->

- [1. Overview](#1-overview)
- [2. Essential Commands](#2-essential-commands)
  - [2.1. Testing](#21-testing)
  - [2.2. Training](#22-training)
  - [2.3. Dataset Viewer](#23-dataset-viewer)
  - [2.4. Code Quality](#24-code-quality)
- [3. Design Ideas](#3-design-ideas)
  - [3.1. Core Design Philosophy](#31-core-design-philosophy)
  - [3.2. Framework Design Philosophy](#32-framework-design-philosophy)
  - [3.3. Dataset Design Pattern](#33-dataset-design-pattern)
  - [3.4. Data Pipeline Design](#34-data-pipeline-design)
  - [3.5. Asynchronous Buffer Pattern](#35-asynchronous-buffer-pattern)
  - [3.6. Configuration-Driven Architecture](#36-configuration-driven-architecture)
  - [3.7. Single-Task Component API Contracts](#37-single-task-component-api-contracts)
  - [3.8. Multi-Task Learning Architecture](#38-multi-task-learning-architecture)
  - [3.9. Training Loop Architecture](#39-training-loop-architecture)
  - [3.10. Metric DIRECTIONS Requirements](#310-metric-directions-requirements)
  - [3.11. Key Directories and Components](#311-key-directories-and-components)
  - [3.12. Special Utilities](#312-special-utilities)
  - [3.13. C++ Extensions](#313-c-extensions)
- [4. Project-Wide Conventions](#4-project-wide-conventions)
  - [4.1. Tensor Type Assumptions](#41-tensor-type-assumptions)
- [5. About Testing](#5-about-testing)
  - [5.1. Testing Philosophy and Patterns](#51-testing-philosophy-and-patterns)
    - [5.1.1. **Common Test Pattern Taxonomy**](#511-common-test-pattern-taxonomy)
    - [5.1.2. **Test Quality Assessment**](#512-test-quality-assessment)
    - [5.1.3. **Recommended Test Templates**](#513-recommended-test-templates)
    - [5.1.4. **Dummy Data Generation for Tests**](#514-dummy-data-generation-for-tests)
  - [5.2. Testing Implementation Guidelines](#52-testing-implementation-guidelines)
  - [5.3. Testing Focus](#53-testing-focus)
  - [5.4. Critical Testing Patterns for Pylon Components](#54-critical-testing-patterns-for-pylon-components)
- [6. Code Style Guidelines](#6-code-style-guidelines)
  - [6.1. Import Statements](#61-import-statements)
    - [6.1.1. Order](#611-order)
    - [6.1.2. Absolute Imports](#612-absolute-imports)
    - [6.1.3. Unused Imports](#613-unused-imports)
  - [6.2. Type Annotations](#62-type-annotations)
  - [6.3. Documentation Strings](#63-documentation-strings)
  - [6.4. Function and File Organization](#64-function-and-file-organization)
  - [6.5. Error Handling](#65-error-handling)
- [7. Important Implementation Notes](#7-important-implementation-notes)

----------------------------------------------------------------------------------------------------

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 1. Overview

Pylon is a PyTorch-based deep learning framework for computer vision research, supporting both 2D vision tasks (change detection, segmentation, object detection) and 3D vision tasks (point cloud registration, 3D change detection) with extensive multi-task learning capabilities.

First look at the entire repo and all the docs to understand everything. if you need, you can always create simple small quick script to validate your factual claims, when you are unsure about anything and want to confirm. such scripts should be designed to answer your own questions about the code base and should have descriptive outputs that you can read from when you execute them. when you are reading and understanding and checking things, you can always refer to all the docs in @docs/ folder, and the @CLAUDE.md and the @README.md . you can ask me any questions and you can update any doc to reflect your knowledge, if you can improve any of the docs. after that, i want you to ultrathink about the solution to the task and let's have a discussion and agree on an implementation plan first. please maintain a implementation_plan.md as you go so that I know what you are thinking and I can provide my feedback when you are done. take your time.

## 2. Essential Commands

### 2.1. Testing
```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/models/change_detection/test_change_star.py

# Run with verbose output
pytest -v --tb=short tests/
```

**CRITICAL TESTING RULE:** ALWAYS run pytest from the project root directory (`~/repos/Pylon/`). NEVER run pytest from within subdirectories like `tests/debuggers/` or `tests/runners/`. This ensures proper module resolution and import paths.

```bash
# ✅ CORRECT - Always run from project root
pytest tests/debuggers/test_forward_debugger.py
pytest tests/runners/test_checkpoint_indices.py

# ❌ WRONG - Never run from subdirectories
cd tests/debuggers && pytest test_forward_debugger.py
cd tests/runners && pytest test_checkpoint_indices.py
```

**IMPORTANT:** NEVER use the PYTHONPATH environment variable when running commands. This can interfere with Python's module resolution and cause import errors. The project structure handles imports correctly without it

**CRITICAL LOGS FOLDER RULE:** NEVER attempt to modify anything in the `./logs/` folder. This contains experiment outputs, checkpoints, and results that should never be altered. For testing, always use dummy test data and test objects in the `tests/` directory.

**conftest.py Usage:**
- conftest.py files are for pytest fixtures ONLY, not general class definitions
- Classes defined in conftest.py are NOT automatically available in test files
- Only fixtures (functions decorated with `@pytest.fixture`) are auto-discovered
- For test helper classes, either:
  - Define them directly in the test file (recommended for test-specific classes)
  - Import them from actual source modules
  - Define them as fixtures in conftest.py if they need setup/teardown

**Batch Size Requirements for Testing:**
- **Validation/Evaluation batch size**: ALWAYS use batch_size=1 during validation and evaluation
  - This ensures per-datapoint metric tracking works correctly
  - BaseCollator operates correctly with batch_size=1 
  - The framework is optimized for this pattern with efficient parallel evaluation
  - Use BaseCollator for all test cases, not default PyTorch collate_fn

### 2.2. Training
```bash
# Basic training
python main.py --config-filepath configs/examples/linear/config.py

# Debug mode (3 epochs, small batches)
python main.py --config-filepath configs/examples/linear/config.py --debug
```

### 2.3. Dataset Viewer
```bash
# Launch web-based dataset viewer
python -m data.viewer.cli

# Custom host/port
python -m data.viewer.cli --host 0.0.0.0 --port 8050

# Run LOD performance benchmarks
python -m benchmarks.data.viewer.pc_lod synthetic  # Synthetic point cloud benchmarks
python -m benchmarks.data.viewer.pc_lod real      # Real dataset benchmarks
```

### 2.4. Code Quality
```bash
# Format code
black .
isort .

# Lint
flake8 .
```

## 3. Design Ideas

### 3.1. Core Design Philosophy

## ⚠️ CRITICAL CODING PRINCIPLES ⚠️

### **NO DEFENSIVE PROGRAMMING - FAIL FAST AND LOUD**

**THIS IS THE MOST IMPORTANT PRINCIPLE IN THIS CODEBASE:**

- **NEVER add checks for conditions that should never occur** - let the code crash with clear error messages
- **NEVER handle "impossible" cases** - bugs should fail loudly, not be masked  
- **NEVER use try-catch to hide errors** - only use when you expect different behavior in different cases

### **⚠️ CRITICAL LESSON: DEFENSIVE PROGRAMMING HIDES ROOT CAUSES**

**When you encounter an error, your FIRST instinct should be to investigate WHY it's happening, not HOW to handle it.**

**Real Example from this codebase:**
- **Error**: `KeyError: 'name'` in callback receiving `dataset_info`
- **Defensive approach**: ❌ Add `if 'name' not in dataset_info: return fallback`
- **Root cause investigation**: ✅ Found callback parameter order was wrong - receiving camera state instead of dataset info
- **Result**: Defensive programming would have masked a simple but critical bug

**The defensive approach would have "fixed" the symptom while leaving the real bug (parameter ordering) undiscovered, making debugging exponentially harder.**

### **🔍 DEBUGGING METHODOLOGY: ALWAYS INVESTIGATE ROOT CAUSES**

**When you encounter ANY error, follow this process:**

1. **STOP** - Don't immediately add error handling
2. **INVESTIGATE** - Why is this error occurring? What assumptions are being violated?
3. **TRACE** - Follow the data flow to understand the real problem
4. **FIX THE CAUSE** - Address the root issue, not the symptom
5. **VERIFY** - Ensure the fix addresses the fundamental problem

**Example debugging questions to ask:**
- "Why doesn't this dictionary have the expected key?"
- "What is this variable actually containing?"
- "Are the function parameters in the right order?"
- "Is the data coming from the expected source?"

**Never ask:** "How can I handle this error case?"

**❌ WRONG - Defensive Programming:**
```python
# DON'T check for "impossible" conditions
if data is None:
    return {"error": "no data"}
    
if len(buffer) == 0:
    return empty_result()

# DON'T use try-catch to hide unexpected errors  
try:
    result = process(data)
except Exception:
    return None  # WRONG - hides bugs!
```

**✅ CORRECT - Fail Fast with Assertions:**
```python
# Use assertions to enforce contracts - fail fast if violated
def process_tensor(tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    assert isinstance(tensor, torch.Tensor), f"Expected torch.Tensor, got {type(tensor)}"
    assert tensor.numel() > 0, "Tensor must not be empty"
    
    return {
        'mean': tensor.mean(),
        'std': tensor.std(),
        'shape': list(tensor.shape)
    }

# Let it crash if assumptions are violated - this reveals bugs!
result = process(data)  # Will crash if data is None - GOOD!
return summarize(buffer)  # Will crash if buffer empty - GOOD!

# Only use try-catch when you expect different cases
try:
    return func(args, new_parameter=value)
except TypeError as e:
    if "unexpected keyword argument" in str(e):
        return func(args)  # Handle API compatibility case
    else:
        raise  # Re-raise unexpected errors
```

**PHILOSOPHY: Code should enforce contracts through assertions and natural failures, not through defensive handling of invalid states.**

### **🔑 CRITICAL: ALWAYS USE ASSERTIONS FOR INPUT VALIDATION**

**Every function that receives parameters should validate critical assumptions using assertions:**

**✅ MANDATORY Pattern for Input Validation:**
```python
def callback_function(
    param1: Optional[Dict[str, Any]],
    param2: str,
    param3: int
) -> List[Any]:
    """Function that processes input parameters."""
    # ALWAYS validate critical assumptions with assertions
    assert param1 is not None, "param1 must not be None"
    assert param1 != {}, "param1 must not be empty"
    assert 'required_key' in param1, f"param1 must have 'required_key', got keys: {list(param1.keys())}"
    assert isinstance(param2, str), f"param2 must be str, got {type(param2)}"
    assert param3 >= 0, f"param3 must be non-negative, got {param3}"
    
    # Continue with function logic...
```

**Key assertion patterns to use:**
- **None checks**: `assert param is not None, "param must not be None"`
- **Empty checks**: `assert param != {}, "param must not be empty"`
- **Key existence**: `assert 'key' in dict_param, f"dict_param must have 'key', got keys: {list(dict_param.keys())}"`
- **Type validation**: `assert isinstance(param, expected_type), f"param must be {expected_type}, got {type(param)}"`
- **Value ranges**: `assert param >= 0, f"param must be non-negative, got {param}"`
- **Length validation**: `assert len(param) > 0, f"param must not be empty, got length {len(param)}"`

**This pattern ensures:**
- ✅ **Immediate failure** when assumptions are violated
- ✅ **Clear error messages** indicating exactly what went wrong
- ✅ **Root cause discovery** instead of symptom handling
- ✅ **Consistent validation** across the entire codebase

### **🔧 CRITICAL: ALWAYS USE KWARGS FOR FUNCTION CALLS**

**To prevent parameter ordering mistakes, always use keyword arguments for function calls with multiple parameters:**

**✅ MANDATORY Pattern for Function Calls:**
```python
# ALWAYS use kwargs for multi-parameter function calls
result = some_function(
    param1=value1,
    param2=value2,
    param3=value3
)

# Backend function calls
datapoint = registry.viewer.backend.get_datapoint(
    dataset_name=dataset_name,
    index=datapoint_idx,
    transform_indices=selected_indices
)

# Display function calls
display = create_display(
    dataset_type=dataset_type,
    datapoint=datapoint,
    class_labels=class_labels,
    camera_state=camera_state,
    settings=settings
)
```

**❌ WRONG - Positional arguments are error-prone:**
```python
# DON'T use positional arguments - easy to get order wrong
result = some_function(value1, value2, value3)
datapoint = registry.viewer.backend.get_datapoint(dataset_name, datapoint_idx, selected_indices)
display = create_display(dataset_type, datapoint, class_labels, camera_state, settings)
```

**This pattern prevents:**
- ❌ **Parameter ordering bugs** (arguments passed in wrong order)
- ❌ **Silent failures** when parameters are swapped
- ❌ **Debugging difficulties** when function signatures change
- ❌ **Maintenance overhead** when adding new parameters

### 3.2. Framework Design Philosophy
Pylon follows several fundamental design patterns that enable extensible, reproducible, and high-performance computer vision research.
The `models` module is mostly copied from official repos and is meant for integrating official model implementations into Pylon.
Some of the code in the `criteria`, `metrics`, `optimizers`, and `runners` are other-project-specific, meaning that they are also copied from official implementations of other released work.
The main contribution of Pylon are the follows:
1. Pylon provides a unified code design for a wide range of research domains, including classical deep learning 2D vision, change detection, point cloud registration, and multi-task learning.
2. Pylon works hard on providing a well-designed "framework" that is robust and, at the same time, easy to use and adapt to implement specific research ideas, while paying less attention to domain-specific code.
3. Pylon provides robust experiment management system, implemented via the `agent` module.
4. While integrating project-specific code (mainly the project-specific model, criterion, and metric), Pylon maintains them and makes sure that the integrated code from external sources are up-to-date with the advancement of package versions.
5. Pylon provides robust debugging toolkit including data viewer and evaluation results viewer.
6. Pylon includes advanced visualization optimizations like the Level of Detail (LOD) system for point clouds, providing up to 70x performance improvements for large datasets.

### 3.3. Dataset Design Pattern
**Every dataset follows the three-field structure:**
- `inputs`: Dictionary with actual input data (images, point clouds, etc.)
- `labels`: Dictionary with ground truth labels for supervision
- `meta_info`: Dictionary with metadata (paths, sizes, indices, etc.)

**Implementation workflow:**
1. Define `self.annotations` in `_init_annotations()` - list of annotation dictionaries
2. Implement `_load_datapoint()` using `self.annotations` and helper methods
3. Use helper methods like `_load_seg_labels()`, `_load_amodal_masks()` for modular loading
4. `meta_info` = `self.annotations` + lightweight additional information

**CRITICAL Device Handling Rule:**
- **Datasets must NEVER manually handle device transfers** - BaseDataset handles this intelligently
- **Always create tensors on CPU** in `_load_datapoint()` without device parameter
- **Never use `.to(device)` in datasets** - trust the framework's established pattern
- BaseDataset uses spawn method and intelligent device transfer to handle multiprocessing correctly

### 3.4. Data Pipeline Design
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

**CRITICAL Pickle/Serialization Rules:**
- **Never store torch.Generator as instance attributes** - will cause `TypeError: cannot pickle 'torch._C.Generator' object`
- **Create generators locally in `_load_datapoint()`** to avoid multiprocessing pickle issues
- **Use deterministic per-index seeding**: `generator.manual_seed((self.initial_seed or 0) + idx)`
- **Other non-picklable objects to avoid**: Open file handles, database connections, CUDA tensors in certain configs

### 3.5. Asynchronous Buffer Pattern
**Critical for GPU utilization:** Criteria, metrics, and optimizers use asynchronous buffers to prevent blocking training loops:
- Background thread: `threading.Thread(target=self._buffer_worker, daemon=True)`
- Thread-safe queue: `queue.Queue()` for non-blocking data collection
- Lock-protected buffer access: `threading.Lock()` for thread safety
- Async tensor operations: Detach from computation graph and move to CPU in background
- `add_to_buffer()` is non-blocking; `summarize()` waits for queue to empty

**CRITICAL Metric Structure Requirement:**
- **Metric `summarize()` output must have matching keys** between `scores['per_datapoint']` and `scores['aggregated']`
- This requirement is enforced by evaluation viewer assertions: `assert metric_names_aggregated == metric_names_per_datapoint`
- Example: If aggregated has `{"mse": 0.5}`, then per_datapoint must have same keys: `{"mse": [0.4, 0.6]}`

### 3.6. Configuration-Driven Architecture
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

### 3.7. Single-Task Component API Contracts
**Critical API contracts for SingleTaskMetric and SingleTaskCriterion:**
- `SingleTaskMetric._compute_score()` receives pure tensors (y_pred, y_true), not dictionaries
- `SingleTaskCriterion._compute_loss()` receives pure tensors (y_pred, y_true), not dictionaries
- Dictionary unwrapping is handled by the wrapper classes automatically
- Both classes extract single tensor from single-key dictionaries before calling _compute_* methods
- Example: `{'output': tensor}` → `tensor` before calling `_compute_score(tensor, y_true)`
- These are the base components that MultiTask wrappers build upon

### 3.8. Multi-Task Learning Architecture
**Wrapper pattern for orchestrating multiple tasks:**
- `MultiTaskCriterion` uses `torch.nn.ModuleDict` for task-specific criteria
- `MultiTaskMetric` aggregates results from individual task metrics
- `MTLOptimizer` implements gradient manipulation (PCGrad, MGDA, GradNorm)
- Each task component maintains its own buffer and state

### 3.9. Training Loop Architecture
**Sophisticated training orchestration:**
- **Deterministic execution**: Per-epoch seeding with separate train/val/test seeds
- **Continuous epoch numbering**: Maintains count across resumptions and multi-stage training
- **Asynchronous I/O**: Threaded checkpoint saving and validation scoring
- **GPU monitoring**: Real-time resource tracking during training
- **Robust resumption**: Automatic detection and resumption from last checkpoint
- **Component initialization order**: Critical order in BaseTrainer._init_components_()
  - Metric must be initialized before early stopping (early stopping depends on metric.DIRECTIONS)
  - Debugger must be initialized before checkpoint loading
  - Dependencies between components must be carefully managed
- **Validation/Evaluation batch size**: ALWAYS use batch_size=1 during validation and evaluation
  - This ensures per-datapoint metric tracking works correctly
  - BaseCollator operates correctly with batch_size=1 and handles meta_info properly
  - The framework is optimized for this pattern with efficient parallel evaluation

### 3.10. Metric DIRECTIONS Requirements
**CRITICAL: All metrics must have DIRECTIONS attribute for early stopping and model comparison:**
- **Class-level DIRECTIONS**: For metrics with fixed output structure (e.g., `SemanticSegmentationMetric`)
  ```python
  class SimpleMetric(SingleTaskMetric):
      DIRECTIONS = {"mse": -1}  # Lower is better for MSE
  ```
- **Instance-level DIRECTIONS**: For wrapper metrics with dynamic structure (e.g., `MultiTaskMetric`, `HybridMetric`)
  ```python
  def __init__(self, metric_configs):
      self.DIRECTIONS = {}
      for task_name, task_metric in self.task_metrics.items():
          self.DIRECTIONS[task_name] = task_metric.DIRECTIONS
  ```
- **DIRECTIONS values**: Must be `1` (higher is better) or `-1` (lower is better)
- **Key relationship**: DIRECTIONS keys must be a subset of score keys used in comparisons
  - Not all score keys need directions - only those used for early stopping/model comparison
  - Extra score keys beyond DIRECTIONS are ignored during comparison
  - Missing directions for compared keys cause assertion failures with clear error messages
  - This allows metrics to provide rich diagnostic outputs while only requiring directions for optimization-relevant keys
- **Wrapper propagation**: MultiTaskMetric and HybridMetric build DIRECTIONS from component metrics
- See `docs/metrics/metric_directions.md` for complete implementation guide

### 3.11. Key Directories and Components
- `/configs/`: Template-based experiment configurations with automated generation
- `/data/`: Datasets with caching, transforms, collators, and interactive viewer
- `/data/viewer/`: Web-based dataset visualization with LOD system for point clouds
- `/criteria/`: Loss functions with asynchronous buffer pattern
- `/metrics/`: Evaluation metrics with threading and buffer management (ALL require DIRECTIONS attribute)
- `/optimizers/`: Standard and multi-task optimizers with gradient manipulation
- `/runners/`: Training loops with deterministic execution and GPU monitoring
- `/runners/eval_viewer/`: Web-based evaluation result visualization
- `/schedulers/`: Learning rate schedulers with warmup and multi-component support
- `/utils/`: Core utilities including builders, automation, determinism, and monitoring
- `/benchmarks/data/viewer/pc_lod/`: Comprehensive LOD performance benchmarking system

### 3.12. Special Utilities
**Automation and Distributed Training:**
- **SSH connection pooling**: Thread-safe multi-server experiment management
- **GPU monitoring**: Real-time utilization tracking across servers
- **Run status monitoring**: Automatic detection of stuck/failed/finished experiments
- **Config generation**: Automated creation of experiment variants with different seeds

**Determinism and Reproducibility:**
- **Comprehensive seeding**: Per-epoch, per-phase random seed management via `utils.determinism.set_seed()`
- **State preservation**: Robust checkpoint and resumption handling
- **Validation**: Extensive configuration and type checking
- **CRITICAL RULE**: Global seeding (`torch.manual_seed()`, `numpy.random.seed()`, etc.) must ONLY be done through `utils.determinism.set_seed()` in trainer/evaluator classes. Always use local `torch.Generator` objects for dataset-level randomness to avoid interfering with global deterministic state.

### 3.13. C++ Extensions
Some modules require building:
```bash
# GeoTransformer
cd data/collators/geotransformer && python setup.py install && cd ../../..

# Buffer/OverlapPredator  
cd data/collators/buffer/cpp_wrappers && bash compile_wrappers.sh && cd ../../../..
```

## 4. Project-Wide Conventions

### 4.1. Tensor Type Assumptions
**Standardized tensor formats across the framework:**

Pylon enforces strict tensor type conventions validated by `utils/input_checks/` module:

- **Images**: `(C, H, W)` float32 tensors where C=3, batched: `(N, C, H, W)`
  ```python
  # Individual image
  image = torch.randn(3, 224, 224, dtype=torch.float32)
  # Batched images  
  images = torch.randn(8, 3, 224, 224, dtype=torch.float32)
  ```

- **Segmentation masks**: `(H, W)` int64 tensors, batched: `(N, H, W)`
  ```python
  # Individual mask (semantic, binary, amodal, instance)
  mask = torch.randint(0, num_classes, (224, 224), dtype=torch.int64)
  # Batched masks
  masks = torch.randint(0, num_classes, (8, 224, 224), dtype=torch.int64)
  ```

- **Classification labels**: int64 scalars, batched: `(N,)` int64 tensors
  ```python
  # Individual label
  label = torch.tensor(5, dtype=torch.int64)
  # Batched labels
  labels = torch.randint(0, num_classes, (8,), dtype=torch.int64)
  ```

- **Point clouds**: Dictionary format with mandatory 'pos' key
  ```python
  # Individual point cloud
  pc = {'pos': torch.randn(1024, 3), 'feat': torch.randn(1024, 32)}
  # Batched point clouds (concatenated along point dimension)
  pc = {'pos': torch.randn(2048, 3), 'feat': torch.randn(2048, 32)}
  ```

- **Model predictions**: Follow task-specific formats
  - Classification: `(N, C)` float32 for batched, `(C,)` for individual
  - Segmentation: `(N, C, H, W)` float32 for batched, `(C, H, W)` for individual
  - Depth: `(N, 1, H, W)` float32 for batched, `(1, H, W)` for individual

**CRITICAL for testing**: When generating dummy inputs in `tests/` modules, always follow these type assumptions. The input validation will fail otherwise.

## 5. About Testing

### 5.1. Testing Philosophy and Patterns
**Comprehensive testing approach with standardized patterns:**

#### 5.1.1. **Common Test Pattern Taxonomy**

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

#### 5.1.2. **Test Quality Assessment**

**Well-Implemented Examples:**
- `test_confusion_matrix.py`: Comprehensive parametrized tests with edge cases
- `test_focal_loss.py`: Thorough input validation and reference comparison
- `test_point_cloud_ops.py`: Equivalence testing with comprehensive coverage
- `test_geotransformer.py`: Memory usage monitoring and performance testing

**Files Needing Improvement:**
- Dataset tests (`test_air_change_dataset.py`, etc.): Only basic iteration, missing edge cases
- Model tests (`test_dsam_net.py`, etc.): Only forward pass, missing gradient/validation testing
- Transform tests: Limited determinism and edge case coverage

#### 5.1.3. **Recommended Test Templates**

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

#### 5.1.4. **Dummy Data Generation for Tests**

**ALWAYS use the standardized dummy data generators** in `tests/models/utils/dummy_data_generators.py`:

```python
# Correct - uses proper tensor types
from tests.models.utils.dummy_data_generators import (
    generate_change_detection_data,      # (N, 3, H, W) float32 images
    generate_segmentation_labels,        # (N, H, W) int64 labels
    generate_classification_labels,      # (N,) int64 labels
    generate_point_cloud_data,          # Batched format for registration
    generate_point_cloud_segmentation_data,  # Flattened format for segmentation
)

# Test with proper types
images = generate_change_detection_data(batch_size=2, height=64, width=64)
labels = generate_segmentation_labels(batch_size=2, height=64, width=64, num_classes=10)
```

**NEVER manually create dummy tensors without proper dtypes:**
```python
# Wrong - missing dtype specification
torch.randn(2, 3, 224, 224)                     # Should be dtype=torch.float32
torch.randint(0, 10, (2, 224, 224))            # Should be dtype=torch.int64

# Correct - use generators or specify dtypes  
torch.randn(2, 3, 224, 224, dtype=torch.float32)
torch.randint(0, 10, (2, 224, 224), dtype=torch.int64)
```

### 5.2. Testing Implementation Guidelines
**CRITICAL: Use pytest functions only - NO test classes:**
- **Framework**: Use `pytest` with plain functions ONLY
- **NO test classes**: Never use `class Test*` - always write `def test_*()` functions
- **Parametrization**: Use `@pytest.mark.parametrize` for multiple test cases instead of test classes
- **Test organization**: For large test modules, split by test patterns into separate files:
  ```
  tests/criteria/base_criterion/
  ├── test_initialization.py      # Initialization pattern
  ├── test_buffer_management.py   # Threading/async buffer tests
  ├── test_device_handling.py     # Device transfer tests
  ├── test_edge_cases.py          # Error handling and edge cases
  └── test_determinism.py         # Reproducibility tests
  ```
- **NO __init__.py files**: Test directories should NOT contain `__init__.py` files - they are not Python packages

**Examples of correct vs incorrect test patterns:**
```python
# ❌ WRONG - Never use test classes
class TestModelName:
    def test_initialization(self):
        model = ModelName()
        assert model is not None

# ✅ CORRECT - Use plain pytest functions  
def test_model_name_initialization():
    model = ModelName()
    assert model is not None

# ❌ WRONG - Multiple similar tests as separate functions
def test_model_batch_size_1():
    test_with_batch_size(1)

def test_model_batch_size_2():
    test_with_batch_size(2)

# ✅ CORRECT - Use parametrize for multiple test cases
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_model_different_batch_sizes(batch_size):
    model = ModelName()
    input_data = generate_dummy_data(batch_size=batch_size)
    output = model(input_data)
    assert output.shape[0] == batch_size
```

### 5.3. Testing Focus
**All tests in Pylon are for "your implementation"** - code we've written or integrated:
- **Base classes and wrappers**: Comprehensive testing with all 9 patterns
- **Domain-specific models/losses**: Focus on integration and API correctness
  - Test successful execution with dummy inputs
  - Verify basic input/output shapes and types
  - Test gradient flow and device handling
  - No need to verify mathematical correctness against papers

**Note**: We do not write separate tests for "official_implementation" - all integrated code is tested as "your implementation".

### 5.4. Critical Testing Patterns for Pylon Components

**Test Configuration Requirements:**
```python
# ✅ CORRECT - Always use batch_size=1 for validation/evaluation tests
'val_dataloader': {
    'class': torch.utils.data.DataLoader,
    'args': {
        'batch_size': 1,  # REQUIRED for validation/evaluation
        'shuffle': False,
        'collate_fn': {
            'class': BaseCollator,  # REQUIRED - never use default collate_fn
            'args': {},
        },
    }
},

# ❌ WRONG - These patterns will cause test failures
'batch_size': 32,  # Wrong for validation/evaluation
'collate_fn': None,  # Wrong - breaks meta_info handling
```

**Component Testing Dependencies:**
- **Metric classes**: Must have DIRECTIONS attribute before testing
- **Trainer classes**: Initialize metric before early stopping
- **API contracts**: SingleTaskMetric._compute_score receives tensors, not dicts
- **Error fixing**: Fix root causes, don't hide errors with try-except blocks

## 6. Code Style Guidelines

### 6.1. Import Statements

#### 6.1.1. Order
**Always follow this exact order with NO spaces between imports:**

1. `from typing import` - always first
2. Python native packages (`os`, `sys`, `copy`, etc.)
3. External packages (`numpy`, `torch`, `torchvision`, etc.)
4. Project modules using **full file paths** (not module imports)

Example:
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

#### 6.1.2. Absolute Imports
- **Source code**: Use full file paths - `from data.datasets.base_dataset import BaseDataset`
- **Config files**: Use module imports - `from data.datasets import BaseDataset` (user-friendly)
- **Note**: Config files are program-generated, so manual editing is rare

#### 6.1.3. Unused Imports
Make sure you remove unused imports after coding.

### 6.2. Type Annotations
**Always include type annotations for function/method arguments and return types:**
```python
def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
    # implementation

def build_from_config(config: Dict[str, Any], **kwargs: Any) -> Any:
    # implementation
```

### 6.3. Documentation Strings
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

### 6.4. Function and File Organization
**Break down complex code for maintainability:**
- **Long functions**: Break down using helper functions with clear single responsibilities
- **Long files**: Split into multiple files and organize as modules (folders with `__init__.py`)
- **Test organization**: Group tests by patterns/philosophies rather than single large files

### 6.5. Error Handling and Try-Catch Usage
**CRITICAL: Try-catch blocks are frequently misused for defensive programming - AVOID THIS:**

- **DO NOT use try-catch to hide errors that shouldn't happen** - this masks bugs and makes debugging impossible
- **DO NOT add try-catch blocks "just in case"** - if you don't expect an exception, don't catch it
- **ONLY use try-catch when you legitimately expect different behavior in different cases**

**❌ WRONG - Using try-catch for defensive programming:**
```python
# DON'T catch exceptions you don't expect
try:
    model = build_model(config)
except Exception:
    return None  # WRONG - hides configuration bugs!

# DON'T catch to handle "impossible" cases  
try:
    result = buffer[idx]
except IndexError:
    return empty_result()  # WRONG - idx should always be valid!
```

**✅ CORRECT - Only catch expected exceptions:**
```python
# Catch when you expect API differences
try:
    return transform(image, seed=seed)
except TypeError as e:
    if "unexpected keyword argument 'seed'" in str(e):
        return transform(image)  # Handle old API
    else:
        raise  # Re-raise unexpected errors

# Catch for legitimate conditional behavior
try:
    checkpoint = torch.load(path)
except FileNotFoundError:
    checkpoint = None  # File missing is expected case
```

**Examples of legitimate try-except usage:**
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

### **🚨 CRITICAL: CALLBACK PARAMETER ORDERING**

**Dash callbacks are extremely sensitive to parameter order - parameters must match input/state order exactly:**

**❌ WRONG - Parameter order doesn't match decorator:**
```python
@callback(
    inputs=[
        Input('slider', 'value'),           # 1st input
        Input('settings', 'data'),          # 2nd input  
        Input('camera', 'data')             # 3rd input
    ],
    states=[
        State('dataset-info', 'data'),      # 1st state
        State('index', 'value')             # 2nd state
    ]
)
def my_callback(
    slider_val: int,                        # ✅ 1st input
    settings: dict,                         # ✅ 2nd input
    dataset_info: dict,                     # ❌ WRONG - should be 3rd input (camera)
    index: int,                             # ❌ WRONG - should be 2nd state
    camera: dict                            # ❌ WRONG - should be 1st state
):
```

**✅ CORRECT - Parameter order matches decorator:**
```python
def my_callback(
    slider_val: int,                        # 1st input
    settings: dict,                         # 2nd input
    camera: dict,                           # 3rd input
    dataset_info: dict,                     # 1st state
    index: int                              # 2nd state
):
```

**DEBUGGING TIP:** When callbacks receive unexpected data, always check parameter ordering first!

### 6.6. PyTorch Best Practices
**Tensor creation and device placement:**
- **ALWAYS create tensors directly on the target device** - avoid using `.to()` method
- **Bad practice**: `tensor = torch.randn(size).to(device)` - creates on CPU then moves
- **Good practice**: `tensor = torch.randn(size, device=device)` - creates directly on target device
- This is especially critical in data loading pipelines where efficiency matters
- The `.to()` method should only be used when truly necessary (e.g., loading pre-existing tensors from disk)

**DataLoader multiprocessing:**
- **BaseDataset handles multiprocessing automatically** with `torch.multiprocessing.set_start_method('spawn', force=True)`
- **Datasets should create tensors on CPU** in `_load_datapoint()` without device parameter
- **BaseDataset handles device transfer** intelligently - skips CUDA transfer in worker processes, applies it in main process
- **Never manually handle device transfer in datasets** - trust the framework's established pattern

## 7. Important Implementation Notes
- Uses PyTorch 2.0.0 with CUDA 11.8
- Follows OpenMMLab conventions (mmengine, mmcv, mmdet)
- Emphasizes Python-native objects and inheritance for extensibility
- Dictionary-as-tensor operations for flexible nested data structures
- Async logging with thread-safe buffered writes
