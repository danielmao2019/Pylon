# Bi-Temporal Change Detection Datasets Code Structure

## Code structure trees

`data/datasets/change_detection_datasets/bi_temporal/slpccd_dataset.py`

```text
slpccd_dataset.py
├── import os
├── from typing import Any, Dict, Tuple
├── import torch
├── from data.datasets.change_detection_datasets.base_2dcd_dataset import Base2DCDDataset
├── from data.structures.three_d.point_cloud import load_point_cloud
└── class SLPCCDDataset(Base2DCDDataset)
    ├── # Street-level point cloud pairs from two years, delivered at a fixed point count with the neighbourhoods a change-detection model reads them through.
    ├── def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]] [override]
    │   ├── # Runs the pair through load, extract, process, neighbourhood and assembly, in that order.
    │   ├── calls self._load_point_cloud_files(idx)
    │   ├── impls pc_data = the two clouds, the segmentation cloud and the paths they came from  # impls-node-one-step:skip — one step; the "and" names what it is made of
    │   ├── calls self._extract_positions_and_features(pc_data)
    │   ├── impls extracted_data = the coordinates, features and pre-subsampling lengths  # impls-node-one-step:skip — one step; the "and" names what it is made of
    │   ├── calls self._extract_change_map(pc_data, extracted_data['pc_2_xyz'])
    │   ├── impls change_map = the per-point labels it named
    │   ├── impls data_for_processing = extracted_data carrying change_map
    │   ├── calls self._process_point_clouds(data_for_processing)
    │   ├── impls processed_data = the normalized blocks at the fixed point count
    │   ├── impls data_for_neighborhood = processed_data carrying the two pre-subsampling lengths
    │   ├── calls self._compute_neighborhood_info(data_for_neighborhood)
    │   ├── impls neighborhood_data = the two rebuilt clouds and their cross-cloud neighbourhoods  # impls-node-one-step:skip — one step; the "and" names what it is made of
    │   ├── impls all_data = data_for_neighborhood merged with neighborhood_data
    │   ├── calls self._build_input_structure(all_data)
    │   ├── impls inputs = the per-cloud structure it assembled
    │   ├── impls labels = the processed change map under 'change_map'
    │   ├── calls self._prepare_meta_info(idx, pc_data)
    │   └── return  # inputs, labels, and the meta info it prepared
    └── def _load_point_cloud_files(self, idx: int) -> Dict[str, Any]
        ├── # Reads the pair off disk, and the second cloud's segmentation file with it when this pair shipped one.
        ├── impls pc_1_filepath, pc_2_filepath = the two paths this index's annotation names
        ├── impls pc_2_seg_filepath = pc_2_filepath with '.txt' replaced by '_seg.txt'
        ├── impls has_seg_file = whether pc_2_seg_filepath exists
        ├── impls meta_data = {'xyz': {'layout': ('0', '1', '2')}}  # decimal text names none of its own columns, so the dataset states the layout the reader's positional split used to supply, and states no width: text parses as float64, so asking for float32 here would abort rather than narrow
        ├── calls load_point_cloud(pc_1_filepath, meta_data=meta_data)
        ├── calls load_point_cloud(pc_2_filepath, meta_data=meta_data)
        ├── impls pc_1, pc_2 = the two clouds it loaded, their coordinates narrowed to float32  # a load never narrows any more, so the dataset that wants the single-precision width its models train at does the narrowing itself
        ├── impls pc_2_seg = None
        ├── if has_seg_file
        │   ├── impls seg_meta_data = meta_data carrying {'change_map': {'layout': ('6',)}}  # a caller-stated layout is the whole field set over a source that numbers its columns, so the label column the positional split used to pick out is named here or it is not loaded at all
        │   ├── calls load_point_cloud(pc_2_seg_filepath, meta_data=seg_meta_data)
        │   └── impls pc_2_seg = the segmentation cloud it loaded, narrowed the same way
        └── return  # the two clouds, pc_2_seg, has_seg_file, and the two paths
```

`data/datasets/change_detection_datasets/bi_temporal/urb3dcd_dataset.py`

```text
urb3dcd_dataset.py
├── from typing import Any, Dict, Tuple
├── import numpy as np
├── import torch
├── from sklearn.neighbors import KDTree
├── from data.datasets.change_detection_datasets.base_3dcd_dataset import Base3DCDDataset
├── from data.structures.three_d.point_cloud import load_point_cloud
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
└── class Urb3DCDDataset(Base3DCDDataset)
    ├── # Simulated urban point cloud pairs over seven change types, served either whole or as cylinder patches around centers chosen per epoch.
    ├── VERSION_MAP  # version -> the directory, subdirectory and PLY element name that version stores its clouds under
    ├── def _load_datapoint(self, idx: int, max_attempts: int = 10) -> Tuple[Dict[str, PointCloud], Dict[str, torch.Tensor], Dict[str, Any]] [override]
    │   ├── # Routes to the patch or the whole-pair path, which is the choice made at construction.
    │   ├── if self.patched
    │   │   ├── calls self._load_datapoint_patched(idx, max_attempts)
    │   │   └── return  # the patch datapoint it built
    │   └── else
    │       ├── calls self._load_datapoint_whole(idx)
    │       └── return  # the whole-pair datapoint it built
    ├── def _load_datapoint_whole(self, idx: int) -> Tuple[Dict[str, PointCloud], Dict[str, torch.Tensor], Dict[str, Any]]
    │   ├── # Serves a whole pair as a datapoint, which is the two clouds, the change map and the two paths taken out of what the loader returned.
    │   ├── calls self._load_point_cloud_whole(idx)
    │   ├── impls data = the clouds, change map, kdtrees, index ranges, paths and idx that loader returned  # impls-node-one-step:skip — one step; the "and" names what it is made of
    │   ├── impls inputs = data's two clouds under 'pc_1' and 'pc_2'                                        # impls-node-one-step:skip — one step; the "and" names what it is made of
    │   ├── impls labels = data's change map under 'change_map'
    │   ├── impls meta_info = the two paths data came from
    │   └── return  # inputs, labels and meta_info
    └── def _load_point_cloud_whole(self, idx: int) -> Dict[str, Any]
        ├── # Serves the pair whole and centered, its change map read out of the second cloud's 'label_ch' field.
        ├── assert this index's annotation names both cloud paths
        ├── impls files = the two cloud paths that annotation names
        ├── impls the second path printed as the pair being loaded
        ├── impls element_name = the PLY element name VERSION_MAP gives this version
        ├── impls meta_data = {'xyz': {'dtype': 'float32', 'layout': element_name joined to each of 'x', 'y' and 'z' by a dot}, 'feat': {'layout': element_name joined to 'label_ch' by a dot}}  # a multi-element ply names which element's columns form a field for nobody, so the dataset states the element the retired nameInPly and name_feat arguments named, and the width the retired dtype argument asked of the coordinates alone
        ├── calls load_point_cloud(files['pc_1_filepath'], meta_data=meta_data)
        ├── impls pc1_xyz = the coordinates of the cloud it loaded
        ├── impls pc1_features = a ones column of one entry per point, in pc1_xyz's dtype
        ├── calls load_point_cloud(files['pc_2_filepath'], meta_data=meta_data)
        ├── impls pc2_xyz = the coordinates of the cloud it loaded
        ├── impls pc2_features = a ones column of one entry per point, in pc2_xyz's dtype
        ├── impls change_map = the second cloud's feat squeezed  # the layout above assembles 'label_ch' into feat, which is where this dataset keeps its change labels
        ├── impls pc1_xyz recast to float32
        ├── impls pc2_xyz recast to float32
        ├── impls change_map recast to int64
        ├── calls self._normalize(pc1_xyz, pc2_xyz)  # -> pc1_xyz and pc2_xyz, centered in place
        ├── calls KDTree(np.asarray(pc1_xyz.cpu()), leaf_size=10)
        ├── calls KDTree(np.asarray(pc2_xyz.cpu()), leaf_size=10)
        ├── impls kdtree_1, kdtree_2 = the two trees it built         # a KDTree indexes host memory, so each block is handed over on the cpu
        ├── impls point_idx_pc1 = the index range of pc1_xyz as long  # each point's index is its own position in the whole cloud
        ├── impls point_idx_pc2 = the index range of pc2_xyz as long
        ├── calls PointCloud(xyz=pc1_xyz, data={'feat': pc1_features})
        ├── calls PointCloud(xyz=pc2_xyz, data={'feat': pc2_features})
        ├── impls pc1, pc2 = the two clouds it built
        └── return  # the two clouds, change_map, the two kdtrees, the two index ranges, the two paths and idx
```
