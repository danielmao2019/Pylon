# Bi-Temporal Change Detection Datasets Code Structure

## Code structure trees

`data/datasets/change_detection_datasets/bi_temporal/slpccd_dataset.py`

```text
slpccd_dataset.py
├── from typing import Any, Dict, Tuple
├── import torch
├── from data.datasets.change_detection_datasets.base_2dcd_dataset import Base2DCDDataset
└── class SLPCCDDataset(Base2DCDDataset)
    ├── # Street-level point cloud pairs from two years, delivered at a fixed point count with the neighbourhoods a change-detection model reads them through.
    └── def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]] [override]
        ├── # Runs the pair through load, extract, process, neighbourhood and assembly, in that order.
        ├── calls self._load_point_cloud_files(idx)
        ├── impls pc_data = the two clouds, the segmentation cloud and the paths they came from  # impls-node-one-step:skip — one step; the "and" names what it is made of
        ├── calls self._extract_positions_and_features(pc_data)
        ├── impls extracted_data = the coordinates, features and pre-subsampling lengths  # impls-node-one-step:skip — one step; the "and" names what it is made of
        ├── calls self._extract_change_map(pc_data, extracted_data['pc_2_xyz'])
        ├── impls change_map = the per-point labels it named
        ├── impls data_for_processing = extracted_data carrying change_map
        ├── calls self._process_point_clouds(data_for_processing)
        ├── impls processed_data = the normalized blocks at the fixed point count
        ├── impls data_for_neighborhood = processed_data carrying the two pre-subsampling lengths
        ├── calls self._compute_neighborhood_info(data_for_neighborhood)
        ├── impls neighborhood_data = the two rebuilt clouds and their cross-cloud neighbourhoods  # impls-node-one-step:skip — one step; the "and" names what it is made of
        ├── impls all_data = data_for_neighborhood merged with neighborhood_data
        ├── calls self._build_input_structure(all_data)
        ├── impls inputs = the per-cloud structure it assembled
        ├── impls labels = the processed change map under 'change_map'
        ├── calls self._prepare_meta_info(idx, pc_data)
        └── return  # inputs, labels, and the meta info it prepared
```

`data/datasets/change_detection_datasets/bi_temporal/urb3dcd_dataset.py`

```text
urb3dcd_dataset.py
├── from typing import Any, Dict, Tuple
├── import torch
├── from data.datasets.change_detection_datasets.base_3dcd_dataset import Base3DCDDataset
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
└── class Urb3DCDDataset(Base3DCDDataset)
    ├── # Simulated urban point cloud pairs over seven change types, served either whole or as cylinder patches around centers chosen per epoch.
    ├── VERSION_MAP  # version -> the directory, subdirectory and PLY element name that version stores its clouds under
    └── def _load_datapoint(self, idx: int, max_attempts: int = 10) -> Tuple[Dict[str, PointCloud], Dict[str, torch.Tensor], Dict[str, Any]] [override]
        ├── # Routes to the patch or the whole-pair path, which is the choice made at construction.
        ├── if self.patched
        │   ├── calls self._load_datapoint_patched(idx, max_attempts)
        │   └── return  # the patch datapoint it built
        └── else
            ├── calls self._load_datapoint_whole(idx)
            └── return  # the whole-pair datapoint it built
```
