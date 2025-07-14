# ModelNet40 Dataset Implementation Plan

## **Core Understanding** ✅

**ModelNet40 IS a point cloud registration dataset** that should inherit from `BasePCRDataset`:
- **Self-registration approach**: Same object transformed to create source-target pairs
- **Key innovation**: Strategic cropping AFTER transformation creates realistic partial overlaps
- **Overlap computation**: GeoTransformer uses directional overlap (ref→src coverage)
- **Registration challenge**: Finding correspondences between partial views of the same object

## **Critical Implementation Requirements**

### **1. Overlap Computation** ✅ (Partially Done)
- **Pylon existing**: `compute_pc_iou()` - bidirectional IoU-style overlap
- **Need to add**: `compute_registration_overlap()` - GeoTransformer's directional approach
  ```python
  # GeoTransformer style: fraction of ref points with src neighbors within radius
  overlap = np.mean(nn_distances < positive_radius)
  ```

### **2. Cropping Transforms** 🚧 (To Implement)
**RandomPlaneCrop**:
- Generate random plane normal from unit sphere (spherical coordinates)
- Compute dot product distances from plane
- Keep top K points (largest distances = one side of plane)

**RandomPointCrop**:
- Sample random viewpoint in 3D space
- Compute distances from viewpoint to all points
- Keep nearest K points (closest to viewpoint)

### **3. ModelNet Dataset** 🚧 (To Implement)
- **Inherit from BasePCRDataset** ✅
- **Load OFF files** using utils/io module
- **Follow GeoTransformer's exact logic**:
  1. Load → Normalize → Sample → Split → Transform → Crop → Overlap Check → Retry

### **4. Transform Pipeline Integration** 🚧 (To Implement)
- Integrate cropping into Pylon's transform system
- Handle retry mechanism for overlap filtering
- Maintain deterministic seeding

## **Implementation Steps**

### **Phase 1: Core Infrastructure**
1. ✅ Add OFF file support to `utils/io/point_cloud.py`
2. 🚧 Add `compute_registration_overlap()` to `utils/point_cloud_ops/`
3. 🚧 Implement `RandomPlaneCrop` transform in `data/transforms/vision_3d/`
4. 🚧 Implement `RandomPointCrop` transform in `data/transforms/vision_3d/`

### **Phase 2: Dataset Implementation**
5. 🚧 Create `ModelNet40Dataset` inheriting from `BasePCRDataset`
6. 🚧 Implement GeoTransformer's exact data generation logic
7. 🚧 Add overlap filtering with retry mechanism
8. 🚧 Handle deterministic seeding and reproducibility

### **Phase 3: Testing & Integration**
9. 🚧 Create comprehensive tests
10. 🚧 Test with various overlap ranges and cropping methods
11. 🚧 Integrate with Pylon's existing PCR infrastructure
12. 🚧 Validate against GeoTransformer reference implementation

## **Technical Details**

### **Transform Specifications**
```python
# RandomPlaneCrop - exact GeoTransformer logic
def random_plane_crop(points, keep_ratio=0.7):
    num_samples = int(np.floor(points.shape[0] * keep_ratio + 0.5))
    p_normal = random_sample_plane()  # spherical coordinates
    distances = np.dot(points, p_normal)
    sel_indices = np.argsort(-distances)[:num_samples]  # largest distances
    return points[sel_indices]

# RandomPointCrop - exact GeoTransformer logic  
def random_point_crop(points, keep_ratio=0.7):
    num_samples = int(np.floor(points.shape[0] * keep_ratio + 0.5))
    viewpoint = random_sample_viewpoint()  # random 3D point
    distances = np.linalg.norm(viewpoint - points, axis=1)
    sel_indices = np.argsort(distances)[:num_samples]  # closest points
    return points[sel_indices]
```

### **Overlap Computation**
```python
# GeoTransformer approach - directional overlap
def compute_registration_overlap(ref_points, src_points, transform=None, positive_radius=0.1):
    if transform is not None:
        src_points = apply_transform(src_points, transform)
    nn_distances = get_nearest_neighbor(ref_points, src_points)
    overlap = np.mean(nn_distances < positive_radius)
    return overlap
```

### **Dataset Logic Flow**
```python
# GeoTransformer's exact process
while True:
    # Load and preprocess
    points = load_and_normalize(file_path)
    ref_points, src_points = split_and_transform(points)
    
    # Crop both independently
    ref_points = random_crop(ref_points, keep_ratio=0.7, method='plane')
    src_points = random_crop(src_points, keep_ratio=0.7, method='plane') 
    
    # Check overlap
    overlap = compute_registration_overlap(ref_points, src_points, transform)
    if min_overlap <= overlap <= max_overlap:
        break  # Accept this pair
    # Otherwise retry with new random crops/transforms
```

## **Default Parameters**
- **Overlap range**: 0.3 to 0.8 (30%-80% overlap)
- **Keep ratio**: 0.7 (retain 70% of points after cropping)
- **Rotation magnitude**: 45° 
- **Translation magnitude**: 0.5 units
- **Overlap radius**: 0.05 (5cm threshold)
- **Cropping method**: 'plane' or 'point'

## **Success Criteria**
- ✅ Reproduces GeoTransformer's exact ModelNet behavior
- ✅ Follows Pylon's architectural patterns (BasePCRDataset inheritance)
- ✅ Maintains deterministic reproducibility
- ✅ Integrates seamlessly with existing PCR infrastructure
- ✅ Comprehensive test coverage

**Ready to start Phase 1 implementation!**