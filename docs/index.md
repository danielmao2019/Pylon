# Pylon Documentation

This directory contains documentation for the Pylon deep learning library.

## General Documentation

- [Overview](README.md) - General overview of the Pylon library
- [Environment Setup](environment_setup.md) - How to set up the development environment

## Critical Development Guidelines

**⚠️ MUST READ for all developers:**

All critical coding principles are now integrated into [CLAUDE.md](../CLAUDE.md):
- **Section 3.1**: NO DEFENSIVE PROGRAMMING principle and fail-fast philosophy  
- **Section 3.3**: Dataset device handling and pickle/serialization patterns
- **Section 3.5**: Metric structure requirements for per_datapoint/aggregated key matching
- **Section 6.5**: Error handling and try-catch usage guidelines

## Debugging and Analysis

- [Debuggers Module](debuggers/README.md) - Framework for capturing and visualizing debugging outputs during training and evaluation

## Data Visualization

- [Dataset Viewer](data/viewer/viewer_doc.md) - Interactive web-based dataset visualization tool
- [Viewer Architecture](data/viewer/architecture.md) - Technical architecture and design patterns
- [Level of Detail (LOD) System](data/viewer/lod_system.md) - Performance optimization for large point cloud visualization

## Models

### Change Detection

- [I3PE Model](models/change_detection/i3pe.md) - Information about the I3PE model
- [Siamese KPConv Model](models/change_detection/siamese_kpconv.md) - Information about the Siamese KPConv model
- [FTN Model](models/change_detection/ftn.md) - Information about the FTN model

## Datasets

### Change Detection

- [SLPCCD Dataset](datasets/change_detection/slpccd.md) - Information about the Street-Level Point Cloud Change Detection dataset

## Metrics

- [Implementation Requirements](metrics/implementation_requirements.md) - Critical requirements for metric structure and key matching

## Optimizers

### Multi-Task Optimizers

- [PCGrad Derivation](optimizers/multi_task_optimizers/pcgrad_derivation.md) - Derivation of the PCGrad optimizer
