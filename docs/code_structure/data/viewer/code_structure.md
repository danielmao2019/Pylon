# Data Viewer Code Structure

## 1. Inheritance / type trees

### Backend schemas

```text
class BaseModel
└── class DisplayResponse  # from here down, its complete set of direct subclasses
    ├── class PointDisplayResponse
    │   ├── class ColorPCDisplayResponse
    │   └── class SegmentationPCDisplayResponse
    ├── class PixelDisplayResponse
    │   ├── class ColorImageDisplayResponse
    │   ├── class DepthImageDisplayResponse
    │   ├── class EdgeImageDisplayResponse
    │   ├── class NormalImageDisplayResponse
    │   ├── class SegmentationImageDisplayResponse
    │   └── class InstanceSurrogateImageDisplayResponse
    ├── class VideoDisplayResponse
    ├── class TextDisplayResponse
    ├── class TableDisplayResponse
    ├── class SceneGraphDisplayResponse
    ├── class MeshDisplayResponse
    │   ├── class ColorMeshDisplayResponse
    │   ├── class SegmentationMeshDisplayResponse
    │   ├── class HeatmapMeshDisplayResponse
    │   └── class SparseHeatmapMeshDisplayResponse
    ├── class GaussianDisplayResponse
    │   ├── class ColorGSDisplayResponse
    │   └── class SegmentationGSDisplayResponse
    ├── class CameraDisplayResponse
    ├── class Aabb3dDisplayResponse
    ├── class Aabb2dDisplayResponse
    ├── class PlaceholderDisplayResponse
    └── class LayeredDisplayResponse
```

### Frontend

```text
interface DisplayResponse
├── interface PointDisplayResponse
│   ├── interface ColorPCDisplayResponse
│   └── interface SegmentationPCDisplayResponse
├── interface PixelDisplayResponse
│   ├── interface ColorImageDisplayResponse
│   ├── interface DepthImageDisplayResponse
│   ├── interface EdgeImageDisplayResponse
│   ├── interface NormalImageDisplayResponse
│   ├── interface SegmentationImageDisplayResponse
│   └── interface InstanceSurrogateImageDisplayResponse
├── interface VideoDisplayResponse
├── interface TextDisplayResponse
├── interface TableDisplayResponse
├── interface SceneGraphDisplayResponse
├── interface MeshDisplayResponse
│   ├── interface ColorMeshDisplayResponse
│   ├── interface SegmentationMeshDisplayResponse
│   ├── interface HeatmapMeshDisplayResponse
│   └── interface SparseHeatmapMeshDisplayResponse
├── interface GaussianDisplayResponse
│   ├── interface ColorGSDisplayResponse
│   └── interface SegmentationGSDisplayResponse
├── interface CameraDisplayResponse
├── interface Aabb3dDisplayResponse
├── interface Aabb2dDisplayResponse
├── interface PlaceholderDisplayResponse
└── interface LayeredDisplayResponse
```
