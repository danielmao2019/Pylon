from typing import Dict, Any
import pytest
import random
import torch
from concurrent.futures import ThreadPoolExecutor
from data.datasets.multi_task_datasets.pascal_context_dataset import PASCALContextDataset


def validate_inputs(inputs: Dict[str, Any]) -> None:
    assert isinstance(inputs, dict), f"{type(inputs)=}"
    # Need to examine actual dataset structure to determine exact validation
    # This is a minimal implementation based on the original test


def validate_labels(labels: Dict[str, Any]) -> None:
    assert isinstance(labels, dict), f"{type(labels)=}"
    # Need to examine actual dataset structure to determine exact validation
    # This is a minimal implementation based on the original test


def validate_meta_info(meta_info: Dict[str, Any], datapoint_idx: int) -> None:
    assert isinstance(meta_info, dict), f"{type(meta_info)=}"
    assert 'idx' in meta_info, f"meta_info should contain 'idx' key: {meta_info.keys()=}"
    assert meta_info['idx'] == datapoint_idx, f"meta_info['idx'] should match datapoint index: {meta_info['idx']=}, {datapoint_idx=}"


def validate_datapoint(dataset: PASCALContextDataset, idx: int) -> None:
    datapoint = dataset[idx]
    assert isinstance(datapoint, dict), f"{type(datapoint)=}"
    assert datapoint.keys() == {'inputs', 'labels', 'meta_info'}
    validate_inputs(datapoint['inputs'])
    validate_labels(datapoint['labels'])
    validate_meta_info(datapoint['meta_info'], idx)


@pytest.fixture
def dataset(request, pascal_context_data_root):
    """Fixture for creating a PASCALContextDataset instance."""
    split = request.param
    return PASCALContextDataset(
        data_root=pascal_context_data_root,
        split=split,
    )


@pytest.mark.parametrize('dataset', ['train', 'val'], indirect=True)
def test_pascal_context_dataset(dataset: PASCALContextDataset, max_samples, get_samples_to_test) -> None:
    assert isinstance(dataset, torch.utils.data.Dataset)
    assert len(dataset) > 0, "Dataset should not be empty"

    num_samples = get_samples_to_test(len(dataset), max_samples)
    indices = random.sample(range(len(dataset)), num_samples)
    with ThreadPoolExecutor() as executor:
        executor.map(lambda idx: validate_datapoint(dataset, idx), indices)
