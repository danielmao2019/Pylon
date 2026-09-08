# PCR Datasets Tests Structure

## Tests implementation structure

`tests/data/datasets/pcr_datasets/conftest.py`

```text
conftest.py
├── import pytest
├── from data.datasets.pcr_datasets.kitti_dataset import KITTIDataset
├── from data.datasets.pcr_datasets.threedmatch_dataset import ThreeDMatchDataset, ThreeDLoMatchDataset
├── from data.datasets.pcr_datasets.modelnet40_dataset import ModelNet40Dataset
├── @pytest.fixture
├── def kitti_dataset_config(request, kitti_data_root, use_cpu_device, get_device)
│   ├── # Names the KITTIDataset build config for the split the case parametrized, so a suite states a split rather than a constructor call.
│   ├── calls get_device(use_cpu_device)
│   └── return  # KITTIDataset with its data root, the requested split, and the device it named
├── @pytest.fixture
├── def threedmatch_dataset_config(request, threedmatch_data_root, use_cpu_device, get_device)
│   ├── # Names the ThreeDMatchDataset build config for the parametrized split, at the matching radius the suite validates against.
│   ├── calls get_device(use_cpu_device)
│   └── return  # ThreeDMatchDataset with its data root, the requested split, a matching radius of a tenth, and the device it named
├── @pytest.fixture
├── def threedlomatch_dataset_config(request, threedmatch_data_root, use_cpu_device, get_device)
│   ├── # Names the same config for the low-overlap split, which reads the same data root under a different band.
│   ├── calls get_device(use_cpu_device)
│   └── return  # ThreeDLoMatchDataset with its data root, the requested split, a matching radius of a tenth, and the device it named
├── @pytest.fixture
└── def modelnet40_dataset_config(request, modelnet40_data_root, modelnet40_cache_file, use_cpu_device, get_device)
    ├── # Names the ModelNet40Dataset build config, the case's own parameters merged over the shared root, cache file and device.
    ├── impls dataset_params = a copy of the case's own parameters
    ├── calls get_device(use_cpu_device)
    └── return  # ModelNet40Dataset with the shared root, the cache file, the device it named, and dataset_params merged over them
```

`tests/data/datasets/pcr_datasets/kitti_dataset/test_kitti_dataset.py`

```text
test_kitti_dataset.py
├── import json
├── import random
├── from concurrent.futures import ThreadPoolExecutor
├── from typing import Any, Dict
├── import pytest
├── import torch
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from utils.builders.builder import build_from_config
├── def validate_inputs(inputs: Dict[str, Any]) -> None
│   ├── # Pins what a KITTI datapoint's inputs are: two clouds carrying coordinates and reflectance and nothing else.
│   ├── assert inputs is a dict keyed exactly by src_pc and tgt_pc
│   └── for each of src_pc and tgt_pc
│       ├── assert it is a PointCloud whose fields are exactly xyz and reflectance
│       ├── assert its coordinates are float32 and [N, 3]
│       ├── assert its reflectance is float32 and [N, 1]
│       └── assert the two agree on the point count
├── def validate_labels(labels: Dict[str, Any]) -> None
│   ├── # Pins the label as one float32 [4, 4] transform and nothing else.
│   └── assert labels is a dict keyed exactly by transform, holding a float32 [4, 4] tensor
├── def validate_meta_info(meta_info: Dict[str, Any], datapoint_idx: int) -> None
│   ├── # Pins the meta info as the index the base adds plus the sequence and two frames this dataset names.
│   └── assert it is keyed exactly by idx, seq, t0 and t1, with idx matching datapoint_idx and the frames being ints
└── @pytest.mark.parametrize('kitti_dataset_config', ['train', 'val', 'test'], indirect=True) def test_kitti_dataset(kitti_dataset_config, max_samples, get_samples_to_test)
    ├── # Every split builds a non-empty dataset whose datapoints all carry that structure.
    ├── calls build_from_config(kitti_dataset_config)
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
    ├── impls indices = that many indices sampled at random  # loading every scan pair of a split is far past what a suite can afford, so a sample stands for it
    └── impls validate_datapoint mapped over indices on a ThreadPoolExecutor
```

`tests/data/datasets/pcr_datasets/threedmatch_dataset/test_threedmatch_dataset.py`

```text
test_threedmatch_dataset.py
├── import random
├── from concurrent.futures import ThreadPoolExecutor
├── from typing import Any, Dict
├── import pytest
├── import torch
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from utils.builders.builder import build_from_config
├── def validate_inputs(inputs: Dict[str, Any]) -> None
│   ├── # Pins what a 3DMatch datapoint's inputs are: the two clouds with their unit features, plus the correspondence index pairs.
│   ├── assert inputs is a dict keyed exactly by src_pc, tgt_pc and correspondences
│   ├── assert each cloud's coordinates are float32 and [N, 3], and its feat is a float32 [N, 1] of the same length
│   ├── assert correspondences is an int64 [K, 2]
│   └── if correspondences is non-empty
│       └── assert each column indexes within its own cloud
├── def validate_labels(labels: Dict[str, Any], src_pc: PointCloud, tgt_pc: PointCloud, gt_overlap: float, matching_radius: float = 0.1) -> None
│   ├── # Pins the label as a well-formed rigid transform, and re-measures the overlap it produces against the one the annotation stated.
│   ├── assert labels is a dict keyed exactly by transform, holding a float32 [4, 4] tensor
│   ├── assert its last row is [0, 0, 0, 1] and its rotation block is orthogonal
│   ├── from models.three_d.point_cloud.ops.set_ops.intersection import compute_registration_overlap
│   ├── calls compute_registration_overlap(ref_points=tgt_pc.xyz, src_points=src_pc.xyz, transform=gt_transform, positive_radius=matching_radius)
│   ├── impls recomputed_overlap = the fraction it measured
│   ├── assert recomputed_overlap clears a quarter  # 3DMatch is defined as the pairs overlapping by more than three tenths, so a pair well below that is a broken annotation rather than a hard one
│   ├── assert it sits within a twentieth of gt_overlap
│   └── assert it sits within zero to one
├── def validate_meta_info(meta_info: Dict[str, Any], datapoint_idx: int) -> None
│   ├── # Pins the meta info as the index plus the two paths, the scene, the overlap and the two fragment ids.
│   └── assert it is keyed exactly by those seven, with idx matching datapoint_idx and the overlap a float within zero to one
├── @pytest.mark.parametrize('threedmatch_dataset_config', ['train', 'val', 'test'], indirect=True) def test_threedmatch_dataset(threedmatch_dataset_config, max_samples, get_samples_to_test)
│   ├── # Every 3DMatch split builds a non-empty dataset whose datapoints all carry that structure and that overlap.
│   ├── calls build_from_config(threedmatch_dataset_config)
│   ├── impls dataset = what it built
│   ├── assert dataset is a torch Dataset and non-empty
│   ├── def validate_datapoint(idx: int) -> None [local]
│   │   ├── # Reads one datapoint and puts its three parts through the checks above, at the dataset's own matching radius.
│   │   ├── impls datapoint = dataset[idx]
│   │   ├── assert datapoint is a dict keyed by inputs, labels and meta_info
│   │   ├── calls validate_inputs(datapoint['inputs'])
│   │   ├── calls validate_labels(labels=datapoint['labels'], src_pc=datapoint['inputs']['src_pc'], tgt_pc=datapoint['inputs']['tgt_pc'], gt_overlap=datapoint['meta_info']['overlap'], matching_radius=dataset.matching_radius)
│   │   └── calls validate_meta_info(datapoint['meta_info'], idx)
│   ├── calls get_samples_to_test(len(dataset), max_samples)
│   ├── impls indices = that many indices sampled at random
│   └── impls validate_datapoint mapped over indices on a ThreadPoolExecutor
└── @pytest.mark.parametrize('threedlomatch_dataset_config', ['train', 'val', 'test'], indirect=True) def test_threedlomatch_dataset(threedlomatch_dataset_config, max_samples, get_samples_to_test)
    ├── # The low-overlap split gets the same treatment, since it is the same data under a different band rather than a different datapoint shape.
    ├── calls build_from_config(threedlomatch_dataset_config)
    ├── impls lomatch_dataset = what it built
    ├── assert lomatch_dataset is a torch Dataset and non-empty
    ├── def validate_datapoint(idx: int) -> None [local]
    │   ├── # Reads one datapoint and puts its three parts through the same checks, at that dataset's matching radius.
    │   ├── impls datapoint = lomatch_dataset[idx]
    │   ├── assert datapoint is a dict keyed by inputs, labels and meta_info
    │   ├── calls validate_inputs(datapoint['inputs'])
    │   ├── calls validate_labels(labels=datapoint['labels'], src_pc=datapoint['inputs']['src_pc'], tgt_pc=datapoint['inputs']['tgt_pc'], gt_overlap=datapoint['meta_info']['overlap'], matching_radius=lomatch_dataset.matching_radius)
    │   └── calls validate_meta_info(datapoint['meta_info'], idx)
    ├── calls get_samples_to_test(len(lomatch_dataset), max_samples)
    ├── impls indices = that many indices sampled at random
    └── impls validate_datapoint mapped over indices on a ThreadPoolExecutor
```
