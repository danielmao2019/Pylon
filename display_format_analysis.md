# Pylon Display Format Analysis

Based on analysis of the datapoint display code, here's a comprehensive overview of the current capabilities and what's missing:

## Current Display Capabilities

### 1. **Image Display** (`image_display.py`)
**Supported Formats:**
- RGB images: `[3, H, W]` tensors
- Grayscale images: `[1, H, W]` tensors  
- Batched images: `[1, C, H, W]` (automatically unbatched)
- Multi-channel images: `[C>3, H, W]` (randomly samples 3 channels for RGB display)
- Multiple data types: `uint8`, `float32`, etc.
- Automatic normalization to `[0,1]` range

**Features:**
- Handles extreme values (very large/small)
- Zero-division protection for uniform images
- Statistics calculation with optional change maps
- Supports various Plotly colorscales

### 2. **Segmentation Display** (`segmentation_display.py`)
**Supported Formats:**
- **Tensor format**: `[H, W]` or `[1, H, W]` with class indices (`int64`)
- **Dictionary format**: `{'masks': List[torch.Tensor], 'indices': List[Any]}` for instance segmentation
- Binary masks as boolean tensors
- Multi-class segmentation with automatic color generation
- Interactive class distribution visualization with toggleable bar charts

**Features:**
- Deterministic color generation using golden ratio for visual distinctness
- Class distribution analysis with pixel counts and percentages
- Dash HTML components for rich statistical displays

### 3. **Point Cloud Display** (`point_cloud_display.py`)
**Supported Formats:**
- Point positions: `[N, 3]` tensors
- Optional colors: `[N, 3]` or `[N, C]` tensors
- Optional labels: `[N]` tensors (converted to colors automatically)
- Comprehensive Level-of-Detail (LOD) system with 3 types:
  - **Continuous LOD**: Camera-based adaptive sampling
  - **Discrete LOD**: Pre-computed cached levels
  - **Density LOD**: Percentage-based subsampling

**Features:**
- Browser memory protection (auto-downsample >100K points)
- Multiple LOD algorithms for performance optimization
- Camera state synchronization
- Axis range consistency for multi-view comparison
- Statistics with class distribution analysis

### 4. **Depth Display** (`depth_display.py`)
**Supported Formats:**
- Depth maps: `[H, W]` tensors
- Optional ignore values (set to NaN for visualization)
- Various data types with proper statistical handling

**Features:**
- Proper depth colorscale visualization
- Valid pixel filtering (positive, finite values)
- Comprehensive statistics (min/max/mean/std depth)

### 5. **Edge Display** (`edge_display.py`)
**Supported Formats:**
- Edge maps: `[H, W]` or `[1, H, W]` tensors
- Automatic channel dimension handling
- Multiple data types (int/float)

**Features:**
- Edge percentage calculation (pixels above threshold)
- Type-aware statistical computation
- Grayscale visualization optimized for edge detection

### 6. **Normal Display** (`normal_display.py`)
**Supported Formats:**
- Surface normals: `[3, H, W]` tensors with (x,y,z) components
- Typically in `[-1,1]` range, mapped to `[0,1]` for RGB visualization

**Features:**
- Normal-to-RGB color mapping
- Magnitude-based statistics
- Component-wise range analysis

### 7. **Instance Surrogate Display** (`instance_surrogate_display.py`)
**Supported Formats:**
- Coordinate offsets: `[2, H, W]` tensors
  - Channel 0: Y-offsets to instance centroids
  - Channel 1: X-offsets to instance centroids
- Ignore value handling (default: 250)

**Features:**
- Converts coordinate representation to visualizable pseudo-instance masks
- Magnitude-based quantile binning for visualization
- Comprehensive offset statistics (range, magnitude, std)

## Missing Format Support

### 1. **Object Detection Formats**
- Bounding boxes: `[N, 4]` or `[N, 5]` (with confidence) tensors
- YOLO format: `[N, 5]` (class, cx, cy, w, h)
- Pascal VOC format: `[N, 4]` (xmin, ymin, xmax, ymax)
- COCO format: `[N, 4]` (x, y, width, height)

### 2. **Keypoint Detection Formats**
- Human pose: `[N, K, 3]` (N people, K keypoints, (x,y,visibility))
- Face landmarks: `[N, 68, 2]` or similar
- Object keypoints for 6D pose estimation

### 3. **3D Object Detection**
- 3D bounding boxes: `[N, 7]` or `[N, 9]` (position, size, rotation)
- Oriented bounding boxes with quaternions
- LiDAR detection formats

### 4. **Flow and Motion**
- Optical flow: `[2, H, W]` (u,v displacement vectors)
- Scene flow: `[3, H, W]` for 3D motion
- Motion boundaries and occlusion maps

### 5. **Multi-Scale/Pyramid Formats**
- Feature pyramid networks: `List[torch.Tensor]` at different scales
- Multi-resolution representations
- Hierarchical data structures

### 6. **Uncertainty/Confidence Maps**
- Per-pixel uncertainty: `[H, W]` confidence/variance maps
- Bayesian uncertainty visualization
- Prediction confidence overlays

### 7. **Temporal Data**
- Video sequences: `[T, C, H, W]` tensors
- Temporal point clouds: `[T, N, 3]` sequences
- Time-series visualizations

### 8. **Graph-Based Data**
- Scene graphs: Nodes and edges
- 3D mesh data: Vertices and faces
- Skeleton/tree structures

## Critical Issue: Inconsistent Batch Support

**MAJOR PROBLEM**: The eval viewer needs to display data from the dataloader (post-collation), but most atomic displays don't support batched tensors.

### Current Batch Support Status:

#### **Stackable Tensors** (after BaseCollator `torch.stack(values, dim=0)`):
- **Images**: `[C, H, W]` → `[B, C, H, W]` - ✅ **Partial support** (has bug in double-squeeze)
- **Segmentation**: `[H, W]` → `[B, H, W]` - ❌ **No batch support**
- **Depth**: `[H, W]` → `[B, H, W]` - ❌ **No batch support**
- **Edges**: `[H, W]` → `[B, H, W]` - ❌ **No batch support**
- **Normals**: `[3, H, W]` → `[B, 3, H, W]` - ❌ **No batch support**
- **Instance Surrogate**: `[2, H, W]` → `[B, 2, H, W]` - ❌ **No batch support**

#### **Point Clouds** (special collation):
- **Current**: `[N, 3]` - ✅ **Works correctly** (concatenated/list format, not stacked)

### Required Changes:

**All atomic displays (except point clouds) must support batched inputs** with consistent batch extraction:

```python
def create_X_display(data, title, batch_index=0, **kwargs):
    # Handle batched input from dataloader
    if data.ndim > expected_unbatched_ndim:
        assert batch_index < data.shape[0], f"batch_index {batch_index} >= batch_size {data.shape[0]}"
        data = data[batch_index]  # Extract single sample for visualization
    # Continue with existing unbatched logic...
```

This ensures compatibility with eval viewer's post-collation data while maintaining single-sample visualization capability.