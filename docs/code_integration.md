# Code Integration Workflow <!-- omit in toc -->

## Table of Contents <!-- omit in toc -->

- [1. Overview](#1-overview)
- [Part I: Repository Analysis Guidelines](#part-i-repository-analysis-guidelines)
  - [2. Analysis Phase](#2-analysis-phase)
    - [2.1. Source Repository Locations](#21-source-repository-locations)
    - [2.2. Analysis Deliverables](#22-analysis-deliverables)
    - [2.3. Analysis Validation Checklist](#23-analysis-validation-checklist)
  - [3. Planning Phase](#3-planning-phase)
    - [3.1. Integration Strategy](#31-integration-strategy)
    - [3.2. API Compatibility Requirements](#32-api-compatibility-requirements)
    - [3.3. Planning Deliverables](#33-planning-deliverables)
- [Part II: Integration Implementation Guidelines](#part-ii-integration-implementation-guidelines)
  - [4. Git Branch Setup](#4-git-branch-setup)
  - [5. Commit 1: Original Code Copy](#5-commit-1-original-code-copy)
    - [5.1. File Identification and Copying](#51-file-identification-and-copying)
    - [5.2. Commit Guidelines](#52-commit-guidelines)
  - [6. Commit 2: Import Statement Fixes](#6-commit-2-import-statement-fixes)
    - [6.1. Import Path Updates](#61-import-path-updates)
    - [6.2. Module Registration](#62-module-registration)
  - [7. Commit 3: API Compatibility Changes](#7-commit-3-api-compatibility-changes)
    - [7.1. Model API Updates](#71-model-api-updates)
    - [7.2. Component API Updates](#72-component-api-updates)
  - [8. Commit 4: Test Case Implementation](#8-commit-4-test-case-implementation)
    - [8.1. Test Structure](#81-test-structure)
    - [8.2. Test Patterns](#82-test-patterns)
  - [9. Commit 5: Debug and Fix Implementation](#9-commit-5-debug-and-fix-implementation)
    - [9.1. Test Debugging Process](#91-test-debugging-process)
    - [9.2. Final Validation](#92-final-validation)
- [10. Critical Integration Principles](#10-critical-integration-principles)
  - [10.1. Code Preservation](#101-code-preservation)
  - [10.2. Pylon Framework Alignment](#102-pylon-framework-alignment)
  - [10.3. API Standards](#103-api-standards)

---

## 1. Overview

**Integration Goal**: Integrate external model code into Pylon framework while preserving original implementation logic and ensuring compatibility with Pylon's architecture.

**Success Criteria**:
- Original implementation logic preserved exactly (no algorithmic changes)
- Full API compatibility with Pylon's dataset and training systems
- Proper component registration and configuration integration
- Comprehensive test coverage following Pylon testing patterns

**Integration Approach**: Structured 5-commit workflow for reviewable and organized integration process.

**Prerequisites**:
- Read and understand CLAUDE.md thoroughly
- Familiarize with existing Pylon change detection models for reference patterns
- Have source repositories accessible locally

---

# Part I: Repository Analysis Guidelines

## 2. Analysis Phase

### 2.1. Source Repository Locations
**Note**: Specific repository paths will be provided for each integration task. Example format:
```
/path/to/source/repository/
/path/to/reference/repository/ (if applicable)
```

### 2.2. Analysis Deliverables

**CRITICAL**: Complete thorough line-by-line analysis before any implementation. Create detailed documentation for each component.

#### 2.2.1. Dataset Analysis
**Objective**: Document exact data structures and formats used by the source implementation.

**Required Documentation**:
1. **Benchmark datasets identification**:
   - List all datasets used in experiments
   - Document dataset names, sources, and paper references
   - Identify train/validation/test splits

2. **Data structure specification**:
   ```python
   # Example format for documentation
   datapoint_structure = {
       "inputs": {
           "image1": "torch.Tensor, dtype=torch.float32, shape=(3, H, W)",
           "image2": "torch.Tensor, dtype=torch.float32, shape=(3, H, W)",
           # Document ALL input fields with exact specifications
       },
       "labels": {
           "change_mask": "torch.Tensor, dtype=torch.int64, shape=(H, W)",
           # Document ALL label fields with exact specifications
       },
       "meta_info": {
           # Document metadata fields
       }
   }
   ```

3. **Input preprocessing pipeline**:
   - Document exact normalization values, transformations
   - Identify image resizing strategies and target sizes
   - Note any special preprocessing steps

#### 2.2.2. Model Analysis
**Objective**: Document model architecture, inputs, outputs, and internal processing.

**Required Documentation**:
1. **Model output specification**:
   ```python
   # Example format
   model_outputs = {
       "logits": "torch.Tensor, dtype=torch.float32, shape=(N, C, H, W)",
       # Document ALL output tensors with exact shapes and meanings
   }
   ```

2. **Classification type determination**:
   - **Binary change detection**: Single channel output, BCE loss compatibility
   - **Multi-class change detection**: C channels output, CE loss compatibility
   - Document whether outputs are raw logits or normalized probabilities
   - Identify expected number of classes

3. **Model architecture components**:
   - Document all custom layers, blocks, and modules
   - Identify external dependencies and their versions
   - Map model components to source files

#### 2.2.3. Training Components Analysis
**Objective**: Document loss functions, metrics, and training procedures.

**Required Documentation**:
1. **Loss functions (criteria)**:
   - Identify all loss functions used
   - Document loss combination strategies and weights
   - Note any custom loss implementations

2. **Evaluation metrics**:
   - List all metrics used for evaluation
   - Document metric calculation methods
   - Identify primary metrics for model comparison

3. **Training configuration**:
   - Document hyperparameters, learning rates, schedules
   - Identify optimization strategies
   - Note any special training procedures

### 2.3. Analysis Validation Checklist

- [ ] All datasets identified with exact data structure documentation
- [ ] Model input/output specifications documented with precise tensor shapes and dtypes
- [ ] Classification type (binary vs multi-class) definitively determined
- [ ] All training components (losses, metrics) cataloged
- [ ] External dependencies identified and version requirements noted
- [ ] Preprocessing pipelines completely documented

## 3. Planning Phase

### 3.1. Integration Strategy

**Create detailed implementation plan addressing**:
1. **Component placement strategy**:
   - Which code goes in `models/change_detection/[model_name]/`
   - Which components need separate modules (criteria, metrics, utils)
   - Dependency management approach

2. **API adaptation requirements**:
   - Dataset input format alignment with Pylon conventions
   - Model forward pass signature modifications needed
   - Output format standardization requirements

### 3.2. API Compatibility Requirements

#### 3.2.1. Dataset API Compliance
**Study existing patterns**:
```bash
# Required reading before implementation
data/datasets/change_detection_datasets/
```

**Ensure compliance with**:
- Three-field structure: `inputs`, `labels`, `meta_info`
- Tensor type assumptions from CLAUDE.md section 4.1
- BaseDataset inheritance and method implementation

#### 3.2.2. Model API Compliance
**Study existing patterns**:
```bash
# Required reading before implementation
models/change_detection/__init__.py
models/change_detection/*/
```

**Ensure compliance with**:
- Forward pass signature: `forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]`
- Raw logits output (no probability normalization)
- No loss/metric computation within model class

### 3.3. Planning Deliverables

**Create before implementation**:
1. **`implementation_plan.md`**: Detailed step-by-step integration plan
2. **Component mapping document**: Which source files map to which Pylon locations
3. **API modification list**: Exact changes needed for Pylon compatibility
4. **Testing strategy**: Test patterns to implement for each component

**Planning validation**: Review plan with user before proceeding to implementation phase.

---

# Part II: Integration Implementation Guidelines

**🎯 Implementation Approach**: Work independently on clear tasks, but ask for high-level guidance when confused or stuck to prevent going down wrong paths.

## 4. Git Branch Setup

**Create development branch for integration**:
```bash
git checkout -b integration/[model_name]
```

**Commit Strategy**: 5 structured commits for organized review process:
1. **Commit 1**: Original code copy (no modifications)
2. **Commit 2**: Import statement fixes only
3. **Commit 3**: API compatibility changes only
4. **Commit 4**: Test case implementation
5. **Commit 5**: Debugging and fixes to make tests pass

## 5. Commit 1: Original Code Copy

### 5.1. File Identification and Copying

**Step 1: Trace Import Dependencies**
Starting from the main model entry point, trace through all import statements to identify necessary files:

```bash
# Practical workflow
# 1. Find main model entry point (usually in README or main script)
# 2. Open main model file and list all `from .` or `from local_module` imports
# 3. For each local import, open that file and repeat
# 4. Continue recursively until all dependencies mapped
# 5. Create comprehensive list of all required files
```

**Concrete steps**:
1. Start with main model class file
2. Use `grep -n "^from \." file.py` to find relative imports
3. Use `grep -n "^from [^torch|^from typing]" file.py` to find potential local imports
4. Build dependency tree systematically

**Step 2: Strategic File Copying**
Copy files to appropriate Pylon modules based on their function:

**Create destination directories first**:
```bash
mkdir -p models/change_detection/[model_name]
mkdir -p models/change_detection/[model_name]/layers
mkdir -p models/change_detection/[model_name]/utils
mkdir -p configs/common/models/change_detection/[model_name]
```

**Copy files systematically**:
```bash
# Models and related components
cp /source/repo/model.py models/change_detection/[model_name]/
cp /source/repo/layers/* models/change_detection/[model_name]/layers/
cp /source/repo/utils/* models/change_detection/[model_name]/utils/

# Loss functions (if separate from model)
cp /source/repo/losses.py criteria/

# Metrics (if separate from model)  
cp /source/repo/metrics.py metrics/

# Datasets (if needed)
cp /source/repo/dataset.py data/datasets/change_detection_datasets/
```

**Verify copy operations**:
```bash
# Check all files copied correctly
find models/change_detection/[model_name] -type f -name "*.py" | wc -l
```

**Step 3: Configuration Files**
```bash
# Copy original configs
cp /source/repo/configs/* configs/common/models/change_detection/[model_name]/
```

### 5.2. Commit Guidelines

**Commit Message Format**:
```
[Integration] Add original [model_name] code from official repository

- Direct copy of all necessary files with no modifications
- Files copied from: [source_repo_url]
- Commit hash: [original_commit_hash]
- Components copied:
  - Model: models/change_detection/[model_name]/
  - Criteria: criteria/[criterion_name].py (if applicable)
  - Metrics: metrics/[metric_name].py (if applicable)
  - Configs: configs/common/models/change_detection/[model_name]/
```

**CRITICAL**: This commit should contain ZERO modifications to the copied code. Only file placement changes.

## 6. Commit 2: Import Statement Fixes

### 6.1. Import Path Updates

**Update all import statements** to work with Pylon's module structure:

```python
# Original imports (example)
from .layers import ConvBlock
from utils.helpers import some_function

# Updated for Pylon structure  
from models.change_detection.model_name.layers import ConvBlock
from models.change_detection.model_name.utils.helpers import some_function
```

**External vs Internal Imports**:
- Keep external library imports unchanged
- Update only internal project imports to match Pylon paths
- Follow CLAUDE.md section 6.1 import ordering

### 6.2. Import Verification with Test Files

**Create temporary test files** to systematically verify all imports:

**Key principle**: Test imports in isolation to identify exact issues without running full model code.

**Step 1: Generate import test files**
For each source file, create a temporary test file that executes all its import statements:

```python
# Example: temp_test_model_imports.py
"""
Temporary file to test imports for models/change_detection/model_name/model.py
This file will be deleted after import verification is complete.
"""

# Copy ALL import statements from the original file exactly
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.change_detection.model_name.layers import ConvBlock
from models.change_detection.model_name.utils.helpers import some_function
# ... all other imports from the original file

print("All imports successful for model.py")
```

**Step 2: Test each file systematically**
```bash
# Test imports for each source file
python temp_test_model_imports.py
python temp_test_layers_imports.py  
python temp_test_utils_imports.py
# ... for all source files
```

**Step 3: Fix import issues iteratively**
- Fix imports one file at a time
- Re-run test file after each fix
- Continue until all test files run without errors
- **CRITICAL**: Only fix import paths, do NOT reorder imports or clean up empty lines

**When to ask for guidance**:
- **If confused about import path structure**: "I'm unsure how to structure the import path for [specific case]. What's the correct approach?"
- **If encountering unexpected errors**: "Getting [error type] that I don't understand. Should I investigate [approach A] or [approach B]?"
- **If multiple possible solutions**: "I see several ways to fix this: [list options]. Which direction should I take?"

**Import Fixing Guidelines**:
- **Change ONLY the import paths**: `from .layers` → `from models.change_detection.model_name.layers`
- **Preserve original formatting**: Keep empty lines, comments, and order exactly as in original
- **No cleanup**: Don't remove unused imports or reorganize - this increases diff noise for review

**Example import fix (preserving formatting)**:
```python
# Original file imports (preserve this exact formatting)
from typing import Dict, List

import torch
import torch.nn as nn

from .layers import ConvBlock  # Fix only this line
from .utils.helpers import some_function  # Fix only this line

# After fixing (same formatting, only paths changed)
from typing import Dict, List

import torch
import torch.nn as nn

from models.change_detection.model_name.layers import ConvBlock  
from models.change_detection.model_name.utils.helpers import some_function
```

**Step 4: Document missing dependencies**
When external packages are missing, document them for review:

```
Missing Dependencies Found:
- albumentations==1.3.0 (ImportError in models/model_name/model.py line 15)
- timm==0.9.2 (ImportError in models/model_name/layers/backbone.py line 8)
- segmentation_models_pytorch==0.3.2 (ImportError in models/model_name/utils/encoder.py line 12)

These packages are not listed in docs/environment_setup.md
```

**When to ask for guidance**:
- **If unsure about dependency**: "Package [X] is missing. Should I document it as a new dependency or look for alternatives?"
- **If many missing packages**: "Found [N] missing dependencies. Should I proceed with documenting all or focus on core ones first?"

**Step 5: Clean up test files**
After all imports are fixed and verified:
```bash
# Remove temporary test files
rm temp_test_*_imports.py
```

### 6.3. Module Registration

**Update `__init__.py` files**:

```python
# models/change_detection/__init__.py
from models.change_detection.model_name import ModelClassName

# criteria/__init__.py (if needed)
from criteria.criterion_name import CriterionClassName

# metrics/__init__.py (if needed)  
from metrics.metric_name import MetricClassName
```

**Commit Message Format**:
```
[Integration] Fix import paths for [model_name] integration

- Update all internal import statements for Pylon module structure
- Register components in appropriate __init__.py files
- Import verification: All test files pass without errors
- Dependency verification: [report status - e.g., "All dependencies available" or "Missing packages reported"]
- No functional changes, only import path updates, preserved original formatting
- Temporary test files cleaned up after verification
```

## 7. Commit 3: API Compatibility Changes

### 7.1. Model API Updates

**Forward Pass Signature Changes**:

```python
# Original model forward (example)
def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    # original logic
    return output

# Pylon-compatible forward (minimal change)
def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    img1 = inputs['image1']
    img2 = inputs['image2']
    # original logic unchanged
    return {'logits': output}
```

**Key API Changes**:
- Input signature: Accept `inputs: Dict[str, torch.Tensor]`
- Output format: Return `Dict[str, torch.Tensor]` with raw logits
- Remove loss/metric computations from model forward pass
- Preserve all internal model logic exactly

### 7.2. Component API Updates

**Criteria API (if separate module)**:
- Inherit from `BaseCriterion` (may or may not use `SingleTaskCriterion`)
- Adapt input handling to work with Pylon's criterion interface
- Preserve original loss computation logic

**Metrics API (if separate module)**:
- Inherit from `BaseMetric` (may or may not use `SingleTaskMetric`)
- Add `DIRECTIONS` attribute for Pylon framework compatibility
- Preserve original metric computation logic

**Commit Message Format**:
```
[Integration] Update [model_name] APIs for Pylon compatibility

- Model forward pass: Accept Dict input, return Dict output
- Remove loss/metric computations from model forward pass
- Update component inheritance (BaseCriterion, BaseMetric)
- Add DIRECTIONS attribute to metrics
- Preserve all original computational logic
```

## 8. Commit 4: Test Case Implementation

### 8.1. Test Structure

**Implement comprehensive test suites**:

```python
# tests/models/change_detection/test_[model_name].py
def test_model_initialization()        # Initialization pattern
def test_forward_pass()               # Basic correctness  
def test_input_validation()           # Invalid input pattern
def test_gradient_flow()              # Gradient verification
def test_output_format()              # API compliance testing

# tests/criteria/test_[criterion_name].py (if applicable)
def test_criterion_initialization()
def test_loss_computation()
def test_known_cases()

# tests/metrics/test_[metric_name].py (if applicable)  
def test_metric_initialization()
def test_score_computation()
def test_directions_attribute()
```

**Reference existing tests**: Study similar model tests in `tests/models/change_detection/` for patterns and conventions before writing new tests.

### 8.2. Test Patterns

**Follow Pylon testing patterns** (see CLAUDE.md section 5):
- Use standardized dummy data generators
- Follow pytest function patterns (no test classes)
- Include comprehensive API compliance tests
- Test tensor type compliance

**When to ask for guidance**:
- **If test patterns are unclear**: "I'm unsure how to structure the test for [specific functionality]. Should I follow [pattern A] or [pattern B]?"
- **If API requirements are ambiguous**: "The model returns [X] but tests expect [Y]. Which should I adjust?"

**Commit Message Format**:
```
[Integration] Add comprehensive test suite for [model_name]

- Model tests: initialization, forward pass, gradient flow, API compliance
- Criteria tests: [criterion_name] functionality and integration
- Metrics tests: [metric_name] computation and DIRECTIONS validation
- All tests follow Pylon testing patterns and conventions
```

## 9. Commit 5: Debug and Fix Implementation

### 9.1. Test Debugging Process

**Systematic debugging approach**:
1. **Run tests and identify failures**:
   ```bash
   pytest tests/models/change_detection/test_[model_name].py -v
   ```

2. **Fix issues following fail-fast philosophy**:
   - Investigate root causes, not symptoms
   - Use assertions for input validation
   - No defensive programming or try-catch masking

3. **Validate each fix**:
   - Ensure fix addresses root cause
   - Verify no regression in other tests
   - Maintain original implementation logic

**When to ask for guidance**:
- **If stuck on debugging**: "Test failing with [error type]. I've checked [areas investigated] but unsure of root cause. Should I focus on [approach A] or [approach B]?"
- **If multiple fix options**: "I can fix this by [option 1] or [option 2]. Which aligns better with Pylon patterns?"
- **If uncertain about scope**: "This fix might affect [broader area]. Should I proceed or take a different approach?"

### 9.2. Final Validation

**Before completion**:
- [ ] All model tests pass: `pytest tests/models/change_detection/test_[model_name].py`
- [ ] All component tests pass (criteria, metrics, datasets)
- [ ] Configuration builds model correctly via `build_from_config`
- [ ] All components properly registered and importable
- [ ] No regressions in existing Pylon functionality

**Final verification steps**:
```bash
# Test model can be imported and built
python -c "
from models.change_detection import [ModelClassName]
from utils.builders.builder import build_from_config
from configs.common.models.change_detection.[model_name].base_config import config
model = build_from_config(config)
print('Integration successful - model builds correctly')
"

# Quick regression test on existing functionality
pytest tests/models/change_detection/test_change_star.py::test_change_star_initialization -v
```

**Commit Message Format**:
```
[Integration] Debug and fix [model_name] implementation

- Fix test failures: [list specific issues fixed]
- Address import/dependency issues
- Resolve API compatibility problems  
- All tests now pass successfully
- No changes to original computational logic
```

## 10. Critical Integration Principles

### 10.1. Code Preservation

**NEVER modify**:
- Original algorithmic logic
- Mathematical constants and default parameters  
- Model architecture decisions
- Loss function implementations
- Metric calculation methods

**ONLY modify for Pylon compatibility**:
- Function signatures for API compliance
- Import statements for Pylon module structure
- Device handling (remove manual device management)
- Loss/metric removal from model forward pass

### 10.2. Pylon Framework Alignment

**Must follow**:
- Fail-fast philosophy with assertions (no defensive programming)
- Input validation using assertions (CLAUDE.md section 3.1)
- Kwargs usage for function calls (CLAUDE.md section 3.1)
- Import statement ordering (CLAUDE.md section 6.1)
- Type annotations for all functions (CLAUDE.md section 6.2)

### 10.3. API Standards

**Dataset Integration**:
- Follow three-field structure: `inputs`, `labels`, `meta_info`
- Implement `_load_datapoint()` method correctly
- Use proper tensor dtypes and shapes
- Never handle device transfers manually

**Model Integration**:
- Signature: `forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]`
- Return raw logits, never normalized probabilities
- Remove all loss/metric computations
- Preserve exact model architecture

**Component Integration**:
- Inherit from appropriate Pylon base classes
- Implement required abstract methods
- Follow async buffer patterns where applicable
- Define DIRECTIONS attribute for metrics

---

## Quick Reference Summary

### Phase Checklist:
1. **Analysis Phase**: Understand source code, document data structures, identify dependencies
2. **Planning Phase**: Create implementation plan, map components, define API changes 
3. **Implementation Phase**: Follow 5-commit workflow systematically

### Commit Workflow:
1. **Commit 1**: Copy original files (no modifications)
2. **Commit 2**: Fix imports and register modules
3. **Commit 3**: Update APIs for Pylon compatibility  
4. **Commit 4**: Implement comprehensive tests
5. **Commit 5**: Debug and fix all issues

### Key Principles:
- **Preserve original logic**: Never change algorithmic implementations
- **Minimal API changes**: Only what's needed for Pylon compatibility
- **Ask for guidance**: When stuck or multiple options exist
- **Test thoroughly**: Follow Pylon testing patterns exactly

### Success Criteria:
- All tests pass
- Model integrates with Pylon training pipeline
- Original computational behavior preserved
- Clean, reviewable commit history

**Usage Note**: This document serves as a comprehensive checklist and reference for code integration tasks. Each phase should be completed fully before proceeding to the next, with all deliverables documented and validated.
