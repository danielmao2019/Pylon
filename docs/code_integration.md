# Code Integration Workflow <!-- omit in toc -->

## Table of Contents <!-- omit in toc -->

- [1. Overview](#1-overview)
- [2. Phase 1: Repository Analysis](#2-phase-1-repository-analysis)
  - [2.1. Source Repository Locations](#21-source-repository-locations)
  - [2.2. Analysis Deliverables](#22-analysis-deliverables)
  - [2.3. Analysis Validation Checklist](#23-analysis-validation-checklist)
- [3. Phase 2: Implementation Planning](#3-phase-2-implementation-planning)
  - [3.1. Integration Strategy](#31-integration-strategy)
  - [3.2. API Compatibility Requirements](#32-api-compatibility-requirements)
  - [3.3. Planning Deliverables](#33-planning-deliverables)
- [4. Phase 3: Code Integration](#4-phase-3-code-integration)
  - [4.1. Model Integration](#41-model-integration)
  - [4.2. Component Integration](#42-component-integration)
  - [4.3. Configuration Integration](#43-configuration-integration)
  - [4.4. Integration Validation](#44-integration-validation)
- [5. Phase 4: Testing and Validation](#5-phase-4-testing-and-validation)
  - [5.1. Test Implementation](#51-test-implementation)
  - [5.2. API Compliance Testing](#52-api-compliance-testing)
  - [5.3. Final Validation](#53-final-validation)
- [6. Critical Integration Principles](#6-critical-integration-principles)
  - [6.1. Code Preservation](#61-code-preservation)
  - [6.2. Pylon Framework Alignment](#62-pylon-framework-alignment)
  - [6.3. API Standards](#63-api-standards)

---

## 1. Overview

**Integration Goal**: Integrate CDMaskFormer code from external repositories into Pylon framework while preserving original implementation logic and ensuring compatibility with Pylon's architecture.

**Success Criteria**:
- Original implementation logic preserved exactly (no algorithmic changes)
- Full API compatibility with Pylon's dataset and training systems
- Proper component registration and configuration integration
- Comprehensive test coverage following Pylon testing patterns

## 2. Phase 1: Repository Analysis

### 2.1. Source Repository Locations
```
/home/daniel/repos/Pylon-CDMaskFormer
/home/daniel/repos/rschange
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

## 3. Phase 2: Implementation Planning

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

## 4. Phase 3: Code Integration

### 4.1. Model Integration

#### 4.1.1. Code Preservation Approach
**CRITICAL**: Follow exact copy-paste methodology:

1. **Copy original files exactly** - preserve all logic, constants, defaults
2. **Make ONLY minimal API adaptations** for Pylon compatibility:
   ```python
   # Original signature (example)
   def forward(self, x1, x2):
       return output

   # Pylon-compatible signature (minimal change)
   def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
       x1 = inputs['image1']
       x2 = inputs['image2']
       output = # original logic unchanged
       return {'logits': output}
   ```

3. **Remove only loss/metric computations** from model forward pass
4. **NO additional changes**: No defensive programming, logging, renaming, formatting

#### 4.1.2. File Copy Approach
**Use direct file copying**: Use `cp -r` command to copy entire model directories from official repo, preserving exact structure and organization as intended by original authors.

#### 4.1.3. Component Registration
**Register in**: `models/change_detection/__init__.py`
```python
from models.change_detection.model_name import ModelClassName
```

### 4.2. Component Integration

#### 4.2.1. Criteria Integration
**Location**: `criteria/` (if needed as separate module)
**Requirements**:
- Copy original loss function implementations exactly
- Adapt only for API compatibility if needed (minimal changes)
- May or may not inherit from Pylon base classes depending on original structure
- Follow asynchronous buffer pattern if integrating with Pylon framework

#### 4.2.2. Metrics Integration  
**Location**: `metrics/` (if needed as separate module)
**Requirements**:
- Copy original metric implementations exactly
- Adapt only for API compatibility if needed (minimal changes)
- May or may not inherit from Pylon base classes depending on original structure
- **CRITICAL**: Define `DIRECTIONS` attribute if integrating with Pylon framework (see CLAUDE.md section 3.10)
- Follow metric structure requirements if using Pylon base classes

### 4.3. Configuration Integration

#### 4.3.1. Model Configurations
**Location**: `configs/common/models/change_detection/[model_name]/`
**Requirements**:
- Convert original config format (YAML, JSON, Python, etc.) to Python dict format
- Follow Pylon's `build_from_config` pattern
- Preserve all original hyperparameters exactly

#### 4.3.2. Example Configuration Structure
```python
# configs/common/models/change_detection/model_name/base_config.py
from models.change_detection.model_name import ModelClassName

config = {
    'class': ModelClassName,
    'args': {
        # Exact parameters from original implementation
        'param1': value1,
        'param2': value2,
        # ... preserve all original parameters
    }
}
```

### 4.4. Integration Validation

**Validation Steps**:
1. **Import verification**: All modules import successfully
2. **Model instantiation**: Model creates without errors using configs
3. **Forward pass testing**: Model processes dummy inputs correctly
4. **Output format validation**: Outputs match expected Pylon format

## 5. Phase 4: Testing and Validation

### 5.1. Test Implementation

**Follow Pylon testing patterns** (see CLAUDE.md section 5):

#### 5.1.1. Model Tests
```python
# tests/models/change_detection/test_[model_name].py
def test_model_initialization()        # Initialization pattern
def test_forward_pass()               # Basic correctness
def test_input_validation()           # Invalid input pattern  
def test_gradient_flow()              # Gradient verification
def test_output_format()              # API compliance testing
```

#### 5.1.2. Component Tests
- **Criteria tests**: Follow loss function testing patterns
- **Metrics tests**: Follow metric testing patterns with DIRECTIONS validation
- **Dataset tests**: Follow dataset testing patterns

### 5.2. API Compliance Testing

**Critical API Tests**:
1. **Input format compliance**:
   ```python
   inputs = {'image1': torch.randn(1, 3, 256, 256), 'image2': torch.randn(1, 3, 256, 256)}
   outputs = model(inputs)
   assert isinstance(outputs, dict)
   assert 'logits' in outputs
   ```

2. **Output format validation**:
   ```python
   # Verify raw logits (not probabilities)
   logits = outputs['logits']
   assert not torch.allclose(logits.sum(dim=1), torch.ones(logits.shape[0]))
   ```

3. **Tensor type compliance**: Follow CLAUDE.md section 4.1 requirements

### 5.3. Final Validation

**Before completion**:
- [ ] Relevant model tests pass: `pytest tests/models/change_detection/test_[model_name].py`
- [ ] Model tests complete successfully (initialization, forward pass, gradient flow, etc.)
- [ ] Configuration builds model correctly via `build_from_config`
- [ ] All components properly registered
- [ ] Documentation updated appropriately

## 6. Critical Integration Principles

### 6.1. Code Preservation

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

### 6.2. Pylon Framework Alignment

**Must follow**:
- Fail-fast philosophy with assertions (no defensive programming)
- Input validation using assertions (CLAUDE.md section 3.1)
- Kwargs usage for function calls (CLAUDE.md section 3.1)
- Import statement ordering (CLAUDE.md section 6.1)
- Type annotations for all functions (CLAUDE.md section 6.2)

### 6.3. API Standards

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

**Usage Note**: This document serves as a comprehensive checklist and reference for code integration tasks. Each phase should be completed fully before proceeding to the next, with all deliverables documented and validated.
