from typing import Dict, Any
import pytest
import random
import torch
from concurrent.futures import ThreadPoolExecutor
from data.datasets.semantic_segmentation_datasets.whu_bd_dataset import WHU_BD_Dataset


def validate_inputs(inputs: Dict[str, Any]) -> None:
    assert isinstance(inputs, dict), f"{type(inputs)=}"
    assert set(inputs.keys()) == set(WHU_BD_Dataset.INPUT_NAMES)
    image = inputs['image']
    assert type(image) == torch.Tensor, f"{type(image)=}"
    assert image.ndim == 3 and image.size(0) == 3, f"{image.shape=}"
    assert image.dtype == torch.float32, f"{image.dtype=}"


def validate_labels(labels: Dict[str, Any]) -> None:
    assert isinstance(labels, dict), f"{type(labels)=}"
    assert set(labels.keys()) == set(WHU_BD_Dataset.LABEL_NAMES)
    semantic_map = labels['semantic_map']
    assert type(semantic_map) == torch.Tensor, f"{type(semantic_map)=}"
    assert semantic_map.ndim == 2, f"{semantic_map.shape=}"
    assert semantic_map.dtype == torch.int64, f"{semantic_map.dtype=}"
    assert set(torch.unique(semantic_map).tolist()).issubset({0, 1}), f"{torch.unique(semantic_map)=}"


def validate_meta_info(meta_info: Dict[str, Any], datapoint_idx: int) -> None:
    assert isinstance(meta_info, dict), f"{type(meta_info)=}"
    assert 'idx' in meta_info, f"meta_info should contain 'idx' key: {meta_info.keys()=}"
    assert meta_info['idx'] == datapoint_idx, f"meta_info['idx'] should match datapoint index: {meta_info['idx']=}, {datapoint_idx=}"


def test_whu_bd_sha1sum() -> None:
    _ = WHU_BD_Dataset(
        data_root="./data/datasets/soft_links/WHU-BD", split='train',
        check_sha1sum=True,
    )


@pytest.fixture
def dataset(request):
    """Fixture for creating a WHU_BD_Dataset instance."""
    split = request.param
    return WHU_BD_Dataset(
        data_root="./data/datasets/soft_links/WHU-BD",
        split=split
    )


@pytest.mark.parametrize('dataset', ['train', 'val', 'test'], indirect=True)
def test_whu_bd_dataset(dataset, max_samples, get_samples_to_test):
    assert isinstance(dataset, torch.utils.data.Dataset)

    def validate_datapoint(idx: int) -> None:
        datapoint = dataset[idx]
        assert isinstance(datapoint, dict), f"{type(datapoint)=}"
        assert datapoint.keys() == {'inputs', 'labels', 'meta_info'}
        validate_inputs(datapoint['inputs'])
        validate_labels(datapoint['labels'])
        validate_meta_info(datapoint['meta_info'], idx)

    num_samples = get_samples_to_test(len(dataset), max_samples, default=len(dataset))
    indices = list(range(num_samples))
    with ThreadPoolExecutor() as executor:
        executor.map(validate_datapoint, indices)
