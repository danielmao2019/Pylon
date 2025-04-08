# GeoTransformer Research Notes

## 1. Directory Structure

```
geotransformer/
├── extensions/          # C++ source files
│   ├── common/         # Common utilities
│   │   └── torch_helper.h
│   ├── cpu/           # CPU implementations
│   │   ├── grid_subsampling/
│   │   │   ├── grid_subsampling.cpp
│   │   │   └── grid_subsampling_cpu.cpp
│   │   └── radius_neighbors/
│   │       ├── radius_neighbors.cpp
│   │       └── radius_neighbors_cpu.cpp
│   ├── extra/         # Additional utilities
│   │   ├── cloud/     # Point cloud operations
│   │   └── nanoflann/ # KNN search library
│   └── pybind.cpp     # Python bindings
├── geotransformer/     # Python package directory
│   └── __init__.py
├── grid_subsample.py   # Python wrapper for grid subsampling
├── radius_search.py    # Python wrapper for radius search
├── registration_collate_fn_stack_mode.py  # Main collation function
└── setup.py            # Build configuration
```

## 2. Building and Using the Extensions

This section explains how to build and use the GeoTransformer C++ extensions without modifying your conda environment.

### Building the Extensions

1. **Create Package Directory**
   ```bash
   mkdir -p geotransformer
   touch geotransformer/__init__.py
   ```
   The `__init__.py` file marks the directory as a Python package, allowing Python to import the compiled extensions.

2. **Build the Extensions**
   ```bash
   python setup.py build_ext --inplace
   ```
   This will compile the C++ code and create a `.so` file (Linux/Mac) or `.pyd` file (Windows) in the `geotransformer/` directory.

   Note: You may see a warning about ninja not being found - this is normal and only affects build speed.
   Optional: Install ninja for faster builds with `conda install ninja`

### Using the Extensions

1. **Add to Python Path**
   ```python
   import sys
   sys.path.append('/path/to/geotransformer')
   ```

2. **Import the Extension**
   ```python
   from geotransformer import ext
   ```

### Available Functions

The extension provides optimized implementations for:
- Grid subsampling (`grid_subsampling`)
- Radius neighbor search (`radius_neighbors`)

These functions are used internally by the collation functions for efficient point cloud processing.

### Troubleshooting

If you encounter the error:
```
error: could not create 'geotransformer/ext.cpython-XX-platform.so': No such file or directory
```
Make sure:
1. The `geotransformer/` directory exists
2. The directory contains an empty `__init__.py` file
3. You have write permissions in the directory

## 3. Known Issues and Fixes

### Inf Values in Matching Scores

#### Issue Description
During training, the model may encounter `inf` values in the matching scores tensor, which leads to NaN values during the backward pass. This issue occurs in the `geotransformer.py` file when computing the matching scores using the dot product of feature vectors:

```python
matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats)  # (P, K, K)
```

The problem arises because the feature vectors can have large magnitudes, causing numerical overflow during the dot product computation. This is particularly problematic when the features are not normalized, as their dot product can exceed the maximum representable value in the floating-point format.

#### Solution
To fix this issue, normalize the feature vectors before computing the dot product:

```python
# Normalize feature vectors before computing matching scores
ref_node_corr_knn_feats = F.normalize(ref_node_corr_knn_feats, p=2, dim=-1)
src_node_corr_knn_feats = F.normalize(src_node_corr_knn_feats, p=2, dim=-1)
matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats)  # (P, K, K)
```

This normalization ensures that the feature vectors have unit length, which prevents numerical overflow during the dot product computation. The resulting matching scores will be in the range [-1, 1] before the scaling factor is applied, which is more stable for the Sinkhorn algorithm and prevents the propagation of `inf` values through the network.
