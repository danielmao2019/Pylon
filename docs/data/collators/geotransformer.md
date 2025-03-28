# GeoTransformer C++ Extensions

This document explains how to build and use the GeoTransformer C++ extensions without modifying your conda environment.

## Directory Structure

```
geotransformer/
├── extensions/          # C++ source files
│   ├── cpu/
│   │   ├── grid_subsampling/
│   │   └── radius_neighbors/
│   ├── extra/
│   │   └── cloud/
│   └── pybind.cpp
├── geotransformer/     # Python package directory
│   └── __init__.py
└── setup.py            # Build configuration
```

## Building the Extensions

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

## Using the Extensions

1. **Add to Python Path**
   ```python
   import sys
   sys.path.append('/path/to/geotransformer')
   ```

2. **Import the Extension**
   ```python
   from geotransformer import ext
   ```

## Available Functions

The extension provides optimized implementations for:
- Grid subsampling (`grid_subsampling`)
- Radius neighbor search (`radius_neighbors`)

These functions are used internally by the collation functions for efficient point cloud processing.

## Troubleshooting

If you encounter the error:
```
error: could not create 'geotransformer/ext.cpython-XX-platform.so': No such file or directory
```
Make sure:
1. The `geotransformer/` directory exists
2. The directory contains an empty `__init__.py` file
3. You have write permissions in the directory
