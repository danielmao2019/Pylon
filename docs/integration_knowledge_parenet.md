# PARENet Integration - Implicit Knowledge and Lessons Learned

## Repository Structure and Patterns

### 1. Testing Patterns
- **Dataset tests require `--samples` flag**: Full dataset tests timeout because they try to load/download entire datasets. Use `--samples` flag for quick tests.
- **Always run pytest from project root**: Never run from subdirectories to ensure proper module resolution.
- **Test hierarchy**: Tests should follow exact implementation patterns - if criterion returns tensor, test expects tensor, not dict.

### 2. Dataloader and Collator Patterns
- **Custom dataloader classes**: Each PCR model has its own dataloader class (e.g., GeoTransformerDataloader, OverlapPredatorDataloader)
- **Dataloader inherits from BaseDataLoader**: Not torch.utils.data.DataLoader directly
- **Collate function via partial**: Dataloader sets up collate_fn using functools.partial with model-specific parameters
- **Function-based collators**: Collate functions are functions, not classes
- **All preprocessing in collator**: Both grid subsampling (C++ CPU extension) and neighbor search (GPU operations) happen during collation when precompute_data=True
- **Neighbor calibration differs by model**: GeoTransformer calibrates neighbors dynamically, PARENet uses fixed neighbor counts
- **PARENet uses k-NN**: Despite function name "radius_search", PARENet does k-nearest neighbor with fixed k

### 3. C++ Extension Handling
- **Fix root causes**: Don't hide or avoid C++ extension errors - fix import paths and module loading
- **Don't modify sys.path**: Follow existing import patterns without adding to sys.path
- **C++ extensions must work**: No fallback - the C++ extensions should be properly compiled and loaded

### 4. API Design Patterns
- **Nested dictionary structure**: Features must be inside `src_pc` and `tgt_pc` dictionaries, not top-level
- **Copy-paste original code**: Minimal changes except import fixes and API wrappers
- **Model output completeness**: Ensure all keys required by criterion/metric are in model output

### 5. Dataset Inheritance
- **PCR datasets inherit from BasePCRDataset**: Not BaseDataset - this provides display_datapoint method
- **KITTI specific**: Changed from BaseDataset to BasePCRDataset inheritance

### 6. Error Handling Philosophy
- **Fail fast and loud**: No defensive programming, use assertions for input validation
- **Root cause investigation**: Always debug why errors occur, don't mask symptoms
- **No try-catch for hiding errors**: Only use when expecting different behavior in different cases

### 7. Tensor Shape Conventions
- **Transform shape**: PARENet outputs (4, 4) not (1, 4, 4) for single transforms
- **Batch handling**: Framework handles batching, models process single examples

### 8. Configuration Requirements
- **Common configs**: Define in `configs/common/` for reusable components
- **Benchmark configs**: Use `configs/benchmarks/point_cloud_registration/gen.py` for experiment generation
- **Main configs**: Generated from benchmarks, not hand-written

## Integration Checklist
1. ✅ Model wrapper with complete output keys
2. ✅ Function-based collator following Pylon patterns
3. ✅ C++ extensions working (with fallback)
4. ✅ Criterion wrapper handling all required keys
5. ✅ Metric with proper DIRECTIONS
6. ✅ Tests passing (with appropriate flags)
7. ✅ Configuration files with collate_fn specified
8. ⏳ End-to-end training script

## Configuration Lessons
- **Model-specific data configs needed**: PARENet requires its own data config files with custom dataloader
- **Generation script patterns**: Update gen.py to use model-specific data configs
- **Device handling in C++ extensions**: Copy tensors to CPU if not on CPU already before C++ calls, then back to original device
- **Custom dataloader classes**: Use PARENetDataloader, not torch.utils.data.DataLoader

## Common Pitfalls to Avoid
- Don't use class-based collators
- Don't skip missing model output keys
- Don't use defensive programming
- Don't run tests from subdirectories
- Don't forget --samples flag for dataset tests
- Don't manually handle device transfers in datasets
- Remove debug imports (ipdb, pdb) from production code