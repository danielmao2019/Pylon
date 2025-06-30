# Config Generation System Migration Guide

## Overview

This document describes the migration from Pylon's old string-based config generation system to the new dictionary-based system, including the challenges encountered and solutions implemented.

## Table of Contents

1. [Old vs New Config Systems](#old-vs-new-config-systems)
2. [Key Differences](#key-differences)
3. [Migration Challenges](#migration-challenges)
4. [The Cross-Contamination Bug](#the-cross-contamination-bug)
5. [Validation and Testing](#validation-and-testing)
6. [Best Practices](#best-practices)

## Old vs New Config Systems

### Old System (String-Based)

The old config generation system used string manipulation and imports:

```python
# Old format - uses imports from common configs
config = {
    'runner': None,
    'work_dir': None,
    'epochs': 100,
    # ... basic structure
}

# Then adds imports and updates
from configs.common.datasets.change_detection.train.oscd_data_cfg import data_cfg as train_dataset_config
config.update(train_dataset_config)
```

**Characteristics:**
- Config files import from common configs using `from ... import ...`
- Common configs contain function calls like `transforms_cfg(size=(224, 224))`
- Configs are evaluated at runtime when imported
- More compact file size
- Harder to debug (need to trace imports)

### New System (Dictionary-Based)

The new system uses `ConfigToFile` class to serialize complete config dictionaries:

```python
# New format - fully expanded/hardcoded values
config = {
    'runner': SupervisedSingleTaskTrainer,
    'train_dataset': {
        'class': OSCDDataset,
        'args': {
            'transforms_cfg': {
                'class': Compose,
                'args': {
                    'transforms': [(
                        {
                            'class': RandomCrop,
                            'args': {
                                'size': (224, 224),
                                'resize': None,
                                'interpolation': None,
                            },
                        },
                        [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'change_map')]
                    ), ...]
                }
            }
        }
    }
    # ... all values fully expanded
}
```

**Characteristics:**
- All values are hardcoded/serialized in the file
- No runtime evaluation needed
- Larger file size but completely self-contained
- Easier to debug (what you see is what you get)
- Uses `ConfigToFile` for proper formatting and import generation

## Key Differences

### 1. Import Handling

**Old System:**
```python
from configs.common.datasets.change_detection.val._transforms_cfg import transforms_cfg
# Function is called at runtime
transforms_cfg(size=(224, 224))
```

**New System:**
```python
from data.transforms.compose import Compose
from data.transforms.vision_2d.crop.random_crop import RandomCrop
# Values are already expanded
{
    'class': Compose,
    'args': {
        'transforms': [...]
    }
}
```

### 2. Config Evaluation

- **Old**: Configs are evaluated when imported (dynamic)
- **New**: Configs are pre-evaluated during generation (static)

### 3. File Size

- **Old**: Smaller files (~2KB) due to imports
- **New**: Larger files (~10KB) with all values expanded

## Migration Challenges

### 1. Module Import Contamination

The validation script (`validate_configs.py`) initially had issues with module contamination when loading multiple configs sequentially. This was fixed by clearing `configs.common` modules from `sys.modules` before each config load.

### 2. Transform Parameter Differences

Different models specify different transform parameters for the same dataset:
- Some models use default transforms from common configs
- Others override with specific parameters (e.g., HANet sets `resize=(256, 256)` for OSCD)

### 3. State Preservation

The generation script must handle:
- PyTorch parameters (preserved as references)
- Class references (converted to imports)
- Nested dictionaries and lists
- Special formatting rules

## The Cross-Contamination Bug

### Discovery

When running the validation script, we found that ChangeMamba models had `resize=(256, 256)` in their transforms, even though the generation script didn't explicitly set this value.

### Root Cause

The bug occurred in the generation loop:

```python
# In gen_bi_temporal.py
for dataset, model in itertools.product(datasets, models):
    config = build_config(dataset, model)
    # ...
```

The `build_config` function was doing:
```python
config.update(train_data_cfg)  # Shallow copy!
config.update(val_data_cfg)    # Shallow copy!
```

When HANet (alphabetically before ChangeMamba) modified its transforms:
```python
config['train_dataset']['args']['transforms_cfg'] = transforms_cfg(first='RandomCrop', size=(224, 224), resize=(256, 256))
```

It was modifying the **shared object** from the imported common configs, contaminating it for subsequent models.

### Solution

Use `semideepcopy` to create deep copies of the dataset configs:

```python
# Fixed version
config.update(semideepcopy(train_data_cfg))
config.update(semideepcopy(val_data_cfg))
```

This ensures each model gets its own copy of the config objects, preventing cross-contamination.

## Validation and Testing

### Validation Script (`validate_configs.py`)

The validation script compares backup configs (old format) with current configs (new format) by:
1. Loading both configs into memory
2. Recursively comparing all values
3. Reporting differences in:
   - Values (e.g., `resize=None` vs `resize=(256, 256)`)
   - Types (e.g., class reference vs dict)
   - Structure (e.g., missing/extra keys)

### Common Validation Issues

1. **Transform Length Mismatches**: 
   - Old: 1 validation transform (from function with default params)
   - New: 6 validation transforms (from function with specific params)

2. **Type Differences**:
   - Old: Direct class references (e.g., `SiameseKPConvCollator`)
   - New: Dictionary format with `{'class': ..., 'args': ...}`

3. **Value Differences**:
   - Different parameter values due to model-specific overrides
   - Cross-contamination from sequential generation

## Best Practices

### 1. Always Use Deep Copy

When building configs that will be modified:
```python
config = semideepcopy(base_config)
config.update(semideepcopy(imported_config))
```

### 2. Clear Module Cache

When loading multiple configs sequentially:
```python
# Clear modules that could cause contamination
modules_to_clear = [key for key in sys.modules.keys() 
                   if key.startswith('configs.common')]
for module in modules_to_clear:
    del sys.modules[module]
```

### 3. Verify Generation Output

After regenerating configs:
1. Run the validation script
2. Check specific files for expected values
3. Test that configs can be loaded and used for training

### 4. Document Model-Specific Overrides

Keep track of which models override default transforms:
- HANet, DsferNet, Changer*, ChangeFormer*, ChangeNext*: Custom transforms for different datasets
- ChangeMamba*, DSIFN, TinyCD, etc.: Use default transforms from common configs

### 5. Maintain Backward Compatibility

The new system should produce functionally equivalent configs to the old system, just in a different format. The actual training behavior should remain unchanged.

## Conclusion

The migration from string-based to dictionary-based config generation provides better debugging capabilities and eliminates runtime evaluation complexity. However, it requires careful handling of object references and deep copying to avoid cross-contamination between generated configs. The validation script serves as an essential tool for ensuring the migration maintains correctness across all model and dataset combinations.
