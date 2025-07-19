from typing import Dict, Any
import pytest
import torch
from concurrent.futures import ThreadPoolExecutor
from data.datasets.change_detection_datasets.bi_temporal.levir_cd_dataset import LevirCdDataset


def validate_inputs(inputs: Dict[str, Any]) -> None:
    assert isinstance(inputs, dict), f"{type(inputs)=}"
    assert set(inputs.keys()) == set(LevirCdDataset.INPUT_NAMES)
    img_1 = inputs['img_1']
    img_2 = inputs['img_2']
    assert type(img_1) == torch.Tensor and img_1.ndim == 3 and img_1.dtype == torch.float32
    assert type(img_2) == torch.Tensor and img_2.ndim == 3 and img_2.dtype == torch.float32
    assert img_1.shape == img_2.shape, f"{img_1.shape=}, {img_2.shape=}"


def validate_labels(labels: Dict[str, Any], dataset) -> torch.Tensor:
    assert isinstance(labels, dict), f"{type(labels)=}"
    assert set(labels.keys()) == set(LevirCdDataset.LABEL_NAMES)
    change_map = labels['change_map']
    assert type(change_map) == torch.Tensor and change_map.ndim == 2 and change_map.dtype == torch.int64
    assert set(torch.unique(change_map).tolist()).issubset(set([0, 1])), f"{torch.unique(change_map)=}"
    # Return change_map for class distribution computation
    return change_map


def validate_meta_info(meta_info: Dict[str, Any], datapoint_idx: int) -> None:
    assert isinstance(meta_info, dict), f"{type(meta_info)=}"
    assert 'idx' in meta_info, f"meta_info should contain 'idx' key: {meta_info.keys()=}"
    assert meta_info['idx'] == datapoint_idx, f"meta_info['idx'] should match datapoint index: {meta_info['idx']=}, {datapoint_idx=}"


def validate_class_distribution(class_dist: torch.Tensor, dataset, num_samples: int) -> None:
    # Validate class distribution (only if we processed the full dataset)
    if num_samples == len(dataset):
        assert type(dataset.CLASS_DIST) == list, f"{type(dataset.CLASS_DIST)=}"
        assert class_dist.tolist() == dataset.CLASS_DIST, f"{class_dist=}, {dataset.CLASS_DIST=}"


@pytest.fixture
def dataset(request):
    """Fixture for creating a LevirCdDataset instance."""
    split = request.param
    return LevirCdDataset(
        data_root="./data/datasets/soft_links/LEVIR-CD",
        split=split
    )


@pytest.mark.parametrize('dataset', ['train', 'test', 'val'], indirect=True)
def test_levir_cd_dataset(dataset, max_samples, get_samples_to_test):
    assert isinstance(dataset, torch.utils.data.Dataset)
    assert len(dataset) > 0, "Dataset should not be empty"

    # Class distribution tracking (preserving original test logic)
    class_dist = torch.zeros(size=(dataset.NUM_CLASSES,), device=dataset.device)

    def validate_datapoint(idx: int) -> torch.Tensor:
        datapoint = dataset[idx]
        assert isinstance(datapoint, dict), f"{type(datapoint)=}"
        assert datapoint.keys() == {'inputs', 'labels', 'meta_info'}
        validate_inputs(datapoint['inputs'])
        change_map = validate_labels(datapoint['labels'], dataset)
        validate_meta_info(datapoint['meta_info'], idx)
        return change_map

    num_samples = get_samples_to_test(len(dataset), max_samples)
    indices = list(range(num_samples))

    # Process datapoints and accumulate class distribution
    with ThreadPoolExecutor() as executor:
        change_maps = list(executor.map(validate_datapoint, indices))

    # Accumulate class distribution from all processed change maps
    for change_map in change_maps:
        for cls in range(dataset.NUM_CLASSES):
            class_dist[cls] += torch.sum(change_map == cls)

    # Validate class distribution
    validate_class_distribution(class_dist, dataset, num_samples)
