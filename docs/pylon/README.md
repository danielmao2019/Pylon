# Pylon Framework Documentation

This directory contains documentation for all Pylon framework modules and components.

## 📁 Module Documentation

### 🔧 Configuration System (`configs/`)
- **Template generation** and automated config creation
- **Migration guides** for configuration updates
- **File system integration** for large-scale experiments

### 📊 Data Management (`data/`)
- **[`collators/`](data/collators/)** - Specialized data collation strategies
- **[`datasets/`](datasets/)** - Dataset implementations by research domain
- **[`viewer/`](data/viewer/)** - Interactive web-based dataset visualization

### 🤖 Model Architectures (`models/`)
- **[`change_detection/`](models/change_detection/)** - Temporal change detection models
- **Multi-domain support** - 2D vision, 3D point clouds, and multi-task learning

### 📏 Evaluation System (`metrics/`)
- **Implementation requirements** and API standards
- **[`metric_directions.md`](metrics/metric_directions.md)** - DIRECTIONS attribute specification
- **Multi-task metric aggregation** patterns

### 🔍 Debugging Tools (`debuggers/`)
- **[`design_principles.md`](debuggers/design_principles.md)** - Debugging philosophy
- **[`integration_guide.md`](debuggers/integration_guide.md)** - Framework integration
- **[`testing_guide.md`](debuggers/testing_guide.md)** - Debugging test patterns

### ⚙️ Optimization (`optimizers/`)
- **Standard optimizers** with learning rate scheduling
- **[`multi_task_optimizers/`](optimizers/multi_task_optimizers/)** - MTL-specific optimization

### 🛠️ Core Utilities (`utils/`)
- **[`dynamic_executor.md`](utils/dynamic_executor.md)** - Resource management system
- **Builder patterns** for component instantiation
- **Determinism utilities** for reproducible experiments

### 🎯 Type System
- **[`tensor_conventions.md`](tensor_conventions.md)** - Framework-wide tensor standards
- **Comprehensive type annotations** across all modules
- **API compatibility requirements** for external integrations

## 🏗️ Framework Architecture

### Core Design Principles
1. **Fail-fast philosophy** - Use assertions, avoid defensive programming
2. **Dictionary-based I/O** - Structured data flow with `inputs`, `labels`, `meta_info`
3. **Asynchronous processing** - Non-blocking buffers for GPU utilization
4. **Configuration-driven** - `build_from_config()` pattern for flexibility

### Component Integration
- **BaseDataset inheritance** for all dataset implementations
- **Multi-task wrappers** for scalable multi-objective learning
- **Thread-safe caching** with LRU memory management
- **Device-aware processing** with intelligent CUDA handling

### API Contracts
- **Single-task components** receive pure tensors, not dictionaries
- **Multi-task wrappers** handle dictionary unwrapping automatically
- **DIRECTIONS requirement** for all metrics used in optimization
- **Deterministic execution** with per-epoch seeding

This documentation serves as the authoritative reference for understanding Pylon's architecture, implementing new components, and following established patterns throughout the framework.