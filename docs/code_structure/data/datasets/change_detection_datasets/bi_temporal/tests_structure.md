# Bi-Temporal Change Detection Datasets Tests Structure

## Tests implementation structure

`tests/data/datasets/change_detection_datasets/bi_temporal/slpccd_dataset/test_slpccd_dataset.py`

```text
test_slpccd_dataset.py
├── from concurrent.futures import ThreadPoolExecutor
├── from typing import Any, Dict
├── import pytest
├── import torch
├── from data.datasets.change_detection_datasets.bi_temporal.slpccd_dataset import SLPCCDDataset
├── from utils.builders.builder import build_from_config
├── def validate_inputs(inputs: Dict[str, Any]) -> None
│   ├── # Pins the inputs as exactly the class's own input names, each carrying a coordinate block of three columns.
│   ├── assert inputs is a dict keyed exactly by SLPCCDDataset.INPUT_NAMES
│   └── for each of pc_1 and pc_2
│       ├── assert it carries an xyz tensor whose second dimension is three
│       └── assert that tensor is float32  # decimal text parses as float64 and no load narrows it any more, so this is the width the dataset narrows to itself
├── def validate_labels(labels: Dict[str, Any]) -> None
│   ├── # Pins the labels as exactly the class's own label names, holding a tensor change map.
│   ├── assert labels is a dict keyed exactly by SLPCCDDataset.LABEL_NAMES, holding a torch.Tensor change map
│   └── assert the change map carries one entry per point and is not all one value  # the label column sits past the coordinates under its own index, so a dataset that picked the wrong column would leave a change map of nothing rather than an error
├── def validate_meta_info(meta_info: Dict[str, Any], datapoint_idx: int) -> None
│   ├── # Pins the meta info as carrying the index the base adds and the two source paths.
│   └── assert it holds idx matching datapoint_idx, plus both file paths
└── @pytest.mark.parametrize('dataset_config', ['train', 'val', 'test'], indirect=True) def test_load_real_dataset(dataset_config, max_samples, get_samples_to_test) -> None
    ├── # Every split builds a non-empty dataset whose datapoints all carry that structure.
    ├── calls build_from_config(dataset_config)
    ├── impls dataset = what it built
    ├── assert dataset is a torch Dataset and non-empty
    ├── def validate_datapoint(idx: int) -> None [local]
    │   ├── # Reads one datapoint and puts its three parts through the checks above.
    │   ├── impls datapoint = dataset[idx]
    │   ├── assert datapoint is a dict keyed by inputs, labels and meta_info
    │   ├── calls validate_inputs(datapoint['inputs'])
    │   ├── calls validate_labels(datapoint['labels'])
    │   └── calls validate_meta_info(datapoint['meta_info'], idx)
    ├── calls get_samples_to_test(len(dataset), max_samples)
    ├── impls indices = the leading that many
    └── impls validate_datapoint mapped over indices on a ThreadPoolExecutor
```

`tests/data/datasets/change_detection_datasets/bi_temporal/urb3dcd_dataset/test_urb3dcd_dataset.py`

```text
test_urb3dcd_dataset.py
├── import random
├── from concurrent.futures import ThreadPoolExecutor
├── from typing import Any, Dict
├── import pytest
├── import torch
├── from data.datasets.change_detection_datasets.bi_temporal.urb3dcd_dataset import Urb3DCDDataset
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from utils.builders.builder import build_from_config
├── def validate_point_count_consistency(pc1: PointCloud, change_map: torch.Tensor) -> None
│   ├── # Pins the labels as per point of the cloud they belong to, which is what a patch's own sampling has to preserve.
│   └── assert the cloud's point count equals the change map's length
├── def validate_inputs(inputs: Dict[str, Any]) -> None
│   ├── # Pins the inputs as the two clouds alone, the KD-trees the class names being loading state rather than datapoint content.
│   ├── assert inputs is a dict keyed exactly by pc_1 and pc_2
│   ├── calls validate_point_cloud(inputs['pc_1'], 'pc_1')
│   └── calls validate_point_cloud(inputs['pc_2'], 'pc_2')
├── def validate_labels(labels: Dict[str, Any]) -> None
│   ├── # Pins the labels as carrying a well-formed change map.
│   ├── assert labels is a dict carrying a torch.Tensor change map
│   └── calls validate_change_map(labels['change_map'])
├── def validate_point_cloud(pc: PointCloud, name: str) -> None
│   ├── # Pins one cloud as a PointCloud carrying float coordinates of three columns and a single-column float feature.
│   ├── assert pc is a PointCloud whose xyz is a floating [N, 3]
│   ├── assert its xyz is float32  # the version's element is loaded at whatever width it stores, and the dataset narrows to the width its models train at
│   └── assert it carries a feat field that is a floating [N, 1]
├── def validate_change_map(change_map: torch.Tensor) -> None
│   ├── # Pins the change map as a flat int64 vector whose values all name one of the seven change types.
│   ├── assert it is a one-dimensional torch.long tensor
│   └── assert every distinct value sits within range(Urb3DCDDataset.NUM_CLASSES)
├── def validate_meta_info(meta_info: Dict[str, Any], datapoint_idx: int) -> None
│   ├── # Pins the meta info as the two point index sets, the two source paths, and the index the base adds.
│   ├── assert both point index sets are torch.long tensors
│   ├── assert both file paths are strings
│   └── assert idx matches datapoint_idx
├── @pytest.mark.parametrize('dataset_config', ['train', 'val', 'test'], indirect=True) def test_urb3dcd_dataset(dataset_config, max_samples, get_samples_to_test) -> None
│   ├── # Every split builds a non-empty dataset whose class mapping is self-consistent and whose datapoints all carry that structure.
│   ├── calls build_from_config(dataset_config)
│   ├── impls dataset = what it built
│   ├── assert both label mappings hold NUM_CLASSES entries and invert each other
│   ├── assert the dataset is non-empty
│   ├── def validate_datapoint(idx: int) -> None [local]
│   │   ├── # Reads one datapoint and puts its three parts, and the label-to-point correspondence, through the checks above.
│   │   ├── impls datapoint = dataset[idx]
│   │   ├── calls validate_inputs(datapoint['inputs'])
│   │   ├── calls validate_labels(datapoint['labels'])
│   │   ├── calls validate_point_count_consistency(inputs['pc_2'], labels['change_map'])
│   │   └── calls validate_meta_info(datapoint['meta_info'], idx)
│   ├── calls get_samples_to_test(len(dataset), max_samples)
│   ├── impls indices = the leading that many
│   └── impls validate_datapoint mapped over indices on a ThreadPoolExecutor
├── def test_urb3dcd_dataset_grid_sampling(urb3dcd_data_root, max_samples, get_samples_to_test) -> None
│   ├── # Grid sampling covers the scene systematically rather than drawing per epoch, and its datapoints carry the same structure.
│   ├── calls Urb3DCDDataset(data_root=urb3dcd_data_root, split='train', version=1, patched=True, sample_per_epoch=0, fix_samples=False, radius=50)
│   ├── impls dataset = what it built
│   ├── assert dataset is non-empty
│   ├── calls get_samples_to_test(len(dataset), max_samples)
│   ├── impls indices = that many sampled at random, or the whole range when it is not shorter
│   ├── def validate_datapoint_grid(idx: int) -> None [local]
│   │   ├── # Reads one grid-sampled datapoint and puts it through the same four checks.
│   │   ├── impls datapoint = dataset[idx]
│   │   ├── calls validate_inputs(datapoint['inputs'])
│   │   ├── calls validate_labels(datapoint['labels'])
│   │   ├── calls validate_point_count_consistency(inputs['pc_2'], labels['change_map'])
│   │   └── calls validate_meta_info(datapoint['meta_info'], idx)
│   └── impls validate_datapoint_grid mapped over indices on a ThreadPoolExecutor
├── def test_urb3dcd_dataset_fixed_sampling(urb3dcd_data_root, max_samples, get_samples_to_test) -> None
│   ├── # Fixed sampling draws its centers once at construction, and its datapoints carry the same structure.
│   ├── calls Urb3DCDDataset(data_root=urb3dcd_data_root, split='train', version=1, patched=True, sample_per_epoch=100, fix_samples=True, radius=50)
│   ├── impls dataset = what it built
│   ├── assert dataset is non-empty
│   ├── calls get_samples_to_test(len(dataset), max_samples)
│   ├── impls indices = that many sampled at random, or the whole range when it is not shorter
│   ├── def validate_datapoint_fixed(idx: int) -> None [local]
│   │   ├── # Reads one fixed-sampled datapoint and puts it through the same four checks.
│   │   ├── impls datapoint = dataset[idx]
│   │   ├── calls validate_inputs(datapoint['inputs'])
│   │   ├── calls validate_labels(datapoint['labels'])
│   │   ├── calls validate_point_count_consistency(inputs['pc_2'], labels['change_map'])
│   │   └── calls validate_meta_info(datapoint['meta_info'], idx)
│   └── impls validate_datapoint_fixed mapped over indices on a ThreadPoolExecutor
└── def test_fixed_samples_consistency(urb3dcd_data_root) -> None
    ├── # Fixed sampling is what makes one index name one patch, so reading it twice has to give the same clouds, labels and point indices.
    ├── calls Urb3DCDDataset(data_root=urb3dcd_data_root, sample_per_epoch=100, fix_samples=True)
    ├── impls dataset = what it built
    └── if the dataset is non-empty
        ├── impls datapoint1, datapoint2 = index zero read twice
        ├── assert both clouds' coordinates and features match across the two reads
        ├── assert the two change maps are equal
        └── assert the two paths and both point index sets match
```
