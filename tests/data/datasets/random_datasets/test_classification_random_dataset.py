from typing import Dict, Any
import pytest
import random
import torch
from concurrent.futures import ThreadPoolExecutor
from data.datasets.random_datasets.classification_random_dataset import ClassificationRandomDataset
from utils.input_checks import check_image


def validate_inputs(inputs: Dict[str, Any]) -> None:
    assert isinstance(inputs, dict), f"{type(inputs)=}"
    assert set(inputs.keys()) == set(['image']), f"{inputs.keys()=}"
    check_image(inputs['image'], batched=False)


def validate_labels(labels: Dict[str, Any]) -> None:
    assert isinstance(labels, dict), f"{type(labels)=}"
    assert set(labels.keys()) == set(['target']), f"{labels.keys()=}"
    assert labels['target'].shape == (), f"{labels['target'].shape=}"


def validate_meta_info(meta_info: Dict[str, Any], datapoint_idx: int) -> None:
    assert isinstance(meta_info, dict), f"{type(meta_info)=}"
    assert set(meta_info.keys()) == set(['idx', 'seed']), f"{meta_info.keys()=}"
    assert type(meta_info['seed']) == int, f"{type(meta_info['seed'])=}"
    assert meta_info['idx'] == datapoint_idx, f"meta_info['idx'] should match datapoint index: {meta_info['idx']=}, {datapoint_idx=}"


@pytest.fixture
def dataset(request):
    """Fixture for creating a ClassificationRandomDataset instance."""
    num_classes, num_examples, image_res, initial_seed = request.param
    return ClassificationRandomDataset(num_classes, num_examples, image_res, initial_seed)


@pytest.mark.parametrize("dataset", [
    (10, 1000, (512, 512), None),
    (10, 1000, (512, 512), 0),
], indirect=True)
def test_classification_random_dataset(dataset, max_samples, get_samples_to_test):
    assert isinstance(dataset, torch.utils.data.Dataset)
    assert len(dataset) > 0, "Dataset should not be empty"

    def validate_datapoint(idx: int) -> None:
        datapoint = dataset[idx]
        assert isinstance(datapoint, dict), f"{type(datapoint)=}"
        assert datapoint.keys() == {'inputs', 'labels', 'meta_info'}
        validate_inputs(datapoint['inputs'])
        validate_labels(datapoint['labels'])
        validate_meta_info(datapoint['meta_info'], idx)

    num_samples = get_samples_to_test(len(dataset), max_samples, default=3)
    indices = list(range(num_samples))
    with ThreadPoolExecutor() as executor:
        executor.map(validate_datapoint, indices)
