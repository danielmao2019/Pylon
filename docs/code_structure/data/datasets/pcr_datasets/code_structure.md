# PCR Datasets Code Structure

## Code structure trees

`data/datasets/pcr_datasets/base_pcr_dataset.py`

```text
base_pcr_dataset.py
├── from typing import Any, Dict, Optional, Tuple, Union
├── import plotly.graph_objects as go
├── import torch
├── from data.datasets.base_dataset import BaseDataset
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from data.viewer.utils.displays.points.dash.core_points_display import create_point_cloud_display
├── from models.three_d.point_cloud.ops.set_ops.symmetric_difference import _normalize_points
└── class BasePCRDataset(BaseDataset)
    ├── # The registration-pair base every PCR dataset inherits, which owns the viewer display for one pair rather than any loading path of its own.
    └── @staticmethod def create_union_visualization(src_points: torch.Tensor, tgt_points: torch.Tensor, point_size: float = 2, point_opacity: float = 0.8, camera_state: Optional[Dict[str, Any]] = None, lod_type: str = "continuous", point_cloud_id: Optional[Union[str, Tuple[str, int, str]]] = None, density_percentage: int = 100, axis_ranges: Optional[Dict[str, Tuple[float, float]]] = None, title: str = "Union (Transformed Source + Target)") -> go.Figure
        ├── # Draws both clouds in one view, red against blue, which is how a viewer reads whether the pose aligns them.
        ├── calls _normalize_points(src_points)
        ├── calls _normalize_points(tgt_points)
        ├── impls src_points_normalized, tgt_points_normalized = the two unbatched blocks it returned
        ├── impls union_points = the two blocks concatenated
        ├── impls src_colors = a zeros block of one row per source point with its red channel set to one
        ├── impls tgt_colors = a zeros block of one row per target point with its blue channel set to one
        ├── impls union_colors = the two color blocks concatenated
        ├── calls PointCloud(xyz=union_points, data={'rgb': union_colors})
        ├── impls union_pc = the display cloud it built
        ├── calls create_point_cloud_display(pc=union_pc, color_key=None, highlight_indices=None, title=title, point_size=point_size, point_opacity=point_opacity, camera_state=camera_state, lod_type=lod_type, density_percentage=density_percentage, point_cloud_id=point_cloud_id, axis_ranges=axis_ranges)
        └── return  # the figure it built
```

`data/datasets/pcr_datasets/kitti_dataset.py`

```text
kitti_dataset.py
├── from typing import Any, Dict, Tuple
├── import os
├── import numpy as np
├── import torch
├── import open3d as o3d
├── from data.datasets.pcr_datasets.base_pcr_dataset import BasePCRDataset
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
└── class KITTIDataset(BasePCRDataset)
    ├── # KITTI odometry scan pairs: two velodyne sweeps far enough apart to register, their ground truth refined by ICP.
    └── def _load_datapoint(self, idx: int) -> Tuple[Dict[str, PointCloud], Dict[str, torch.Tensor], Dict[str, Any]] [override]
        ├── # Loads one scan pair and its ground truth, refining that ground truth by ICP the first time the pair is asked for.
        ├── impls ann = self.annotations[idx]
        ├── impls seq = the sequence ann names
        ├── impls t0 = the first frame ann names
        ├── impls t1 = the second frame ann names
        ├── calls self.get_video_odometry(seq, indices=[t0, t1])
        ├── calls self.odometry_to_positions(two_odo)
        ├── impls two_pos = the two poses it built
        ├── impls fname0 = the velodyne file frame t0 names under this sequence
        ├── impls fname1 = the velodyne file frame t1 names under this sequence
        ├── impls xyzr0 = fname0 read as float32 and reshaped to four columns  # impls-node-one-step:skip — one step; the "and" names what it is made of
        ├── impls xyzr1 = fname1 read as float32 and reshaped to four columns  # impls-node-one-step:skip — one step; the "and" names what it is made of
        ├── impls xyz0 = the leading three columns of xyzr0
        ├── impls xyz1 = the leading three columns of xyzr1
        ├── impls icp_cache_file = the cache file this sequence and frame pair names  # impls-node-one-step:skip — one step; the "and" names what it is made of
        ├── if icp_cache_file does not exist
        │   ├── impls M = the odometry-implied source-to-target pose, carried through self.velo2cam at both ends
        │   ├── calls self.apply_transform(xyz0, M)
        │   ├── impls xyz0_t = the seeded source coordinates
        │   ├── impls pcd0 = xyz0_t as a grey open3d cloud
        │   ├── impls pcd1 = xyz1 as a green open3d cloud
        │   ├── impls reg = point-to-point ICP over those two at a fifth-metre threshold, up to two hundred iterations
        │   ├── impls gt_transform = M composed with what ICP returned  # the odometry pose is only a seed, so the pair's stated ground truth is what ICP settles on
        │   └── impls gt_transform saved to icp_cache_file
        ├── else
        │   └── impls gt_transform = what that cache file holds
        ├── calls PointCloud(xyz=torch.from_numpy(xyz0).float().to(self.device), data={'reflectance': the fourth column of xyzr0 as a float32 [N, 1], 'feat': a ones column})
        ├── calls PointCloud(xyz=torch.from_numpy(xyz1).float().to(self.device), data={'reflectance': the fourth column of xyzr1 as a float32 [N, 1], 'feat': a ones column})
        ├── impls src_pc, tgt_pc = the two clouds it built
        └── return  # those two clouds, gt_transform as a float32 label, and the sequence and frames as meta info
```

`data/datasets/pcr_datasets/modelnet40_dataset.py`

```text
modelnet40_dataset.py
├── import os
├── import glob
├── from data.datasets.pcr_datasets.synthetic_transform_pcr_dataset import SyntheticTransformPCRDataset
└── class ModelNet40Dataset(SyntheticTransformPCRDataset)
    ├── # ModelNet40 objects self-registered: one OFF file is loaded as both halves of a pair, and the synthetic base poses and crops them apart.
    └── def _init_annotations(self) -> None [override]
        ├── # Names one annotation per OFF file, both halves of the pair pointing at that same file.
        ├── impls split_dir = self.split
        ├── if self.split is val
        │   └── impls split_dir = test  # ModelNet40 has no val split of its own, so a val run reads the test files
        ├── impls off_files = an empty list
        ├── for each category in self.CATEGORIES
        │   ├── impls category_dir = that category's directory under the data root for split_dir
        │   ├── if category_dir does not exist
        │   │   └── continue
        │   └── impls off_files gains the sorted OFF files under category_dir
        ├── impls self.annotations = an empty list
        ├── for each file_path in off_files
        │   ├── calls self.get_category_from_path(file_path)
        │   └── impls self.annotations gains that file as both source and target, under the category it named  # impls-node-one-step:skip — one step; the "and" names what it is made of
        └── impls the count found printed for the split
```

`data/datasets/pcr_datasets/synthetic_transform_pcr_dataset.py`

```text
synthetic_transform_pcr_dataset.py
├── from abc import ABC
├── from typing import Any, Dict, Tuple
├── import torch
├── from data.datasets.pcr_datasets.base_pcr_dataset import BasePCRDataset
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
└── class SyntheticTransformPCRDataset(BasePCRDataset, ABC)
    ├── # Builds a registration pair out of one cloud by posing and cropping it, trying sampled transforms until the overlap lands in the range asked for.
    └── def _load_datapoint(self, idx: int) -> Tuple[Dict[str, PointCloud], Dict[str, Any], Dict[str, Any]] [override]
        ├── # Loads one synthetic pair, which is whichever trial of this index first produced an overlap in range.
        ├── impls annotation = self.annotations[idx]
        ├── assert annotation is to name both source paths
        ├── assert this subclass implements _apply_crop
        ├── impls t1_pc_filepath, t2_pc_filepath = the two paths that annotation names
        ├── calls self._search_or_generate(t1_pc_filepath=t1_pc_filepath, t2_pc_filepath=t2_pc_filepath, idx=idx)
        ├── impls src_pc, tgt_pc, overlap_ratio, transform_matrix, trial_idx = what it settled on
        ├── if src_pc carries no feat
        │   └── impls src_pc.feat = a float32 ones column of one entry per point
        ├── if tgt_pc carries no feat
        │   └── impls tgt_pc.feat = a float32 ones column of one entry per point
        └── return  # the two clouds, transform_matrix as the label, and the two paths, the trial index, the transform and the overlap as meta info
```

`data/datasets/pcr_datasets/threedmatch_dataset.py`

```text
threedmatch_dataset.py
├── from typing import Any, Dict, Tuple
├── import torch
├── from data.datasets.pcr_datasets.base_pcr_dataset import BasePCRDataset
├── from data.structures.three_d.point_cloud import load_point_cloud
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── class _ThreeDMatchBaseDataset(BasePCRDataset)
│   ├── # The annotated fragment pairs both 3DMatch splits are drawn from, the split itself being an overlap band the subclass names.
│   └── def _load_datapoint(self, idx: int) -> Tuple[Dict[str, PointCloud], Dict[str, torch.Tensor], Dict[str, Any]] [override]
│       ├── # Loads one fragment pair and inverts the annotation's pose, since the metadata states target-to-source and a datapoint states source-to-target.
│       ├── impls annotation = self.annotations[idx]
│       ├── calls load_point_cloud(annotation['src_path'], device=self.device)
│       ├── calls load_point_cloud(annotation['tgt_path'], device=self.device)
│       ├── impls src_pc, tgt_pc = the two clouds it loaded
│       ├── impls each of them gains a float32 ones column as feat
│       ├── impls transform_tgt_to_src = the float32 [4, 4] the annotation's rotation and translation make  # impls-node-one-step:skip — one step; the "and" names what it is made of
│       ├── impls transform = the inverse of transform_tgt_to_src
│       └── return  # the two clouds, transform as the label, and the paths, scene name, overlap and two frame ids as meta info
├── class ThreeDMatchDataset(_ThreeDMatchBaseDataset)
│   ├── # The 3DMatch split of those pairs, which is the ones overlapping by more than three tenths.
│   ├── DATASET_SIZE  # split name -> the pair count that band leaves
│   └── def __init__(self, **kwargs) -> None
│       ├── # Builds the base over this split's overlap band.
│       └── calls super().__init__(overlap_min=0.3, overlap_max=1.0, **kwargs)
└── class ThreeDLoMatchDataset(_ThreeDMatchBaseDataset)
    ├── # The 3DLoMatch split of those same pairs, which is the harder band from a tenth to three tenths.
    ├── DATASET_SIZE  # split name -> the pair count that band leaves
    └── def __init__(self, **kwargs) -> None
        ├── # Builds the base over this split's overlap band.
        └── calls super().__init__(overlap_min=0.1, overlap_max=0.3, **kwargs)
```
