from typing import Dict, Any
import pytest
import random
import torch
import os
from concurrent.futures import ThreadPoolExecutor
from data.datasets.multi_task_datasets.multi_task_facial_landmark_dataset import MultiTaskFacialLandmarkDataset


def validate_inputs(inputs: Dict[str, Any]) -> None:
    assert isinstance(inputs, dict), f"{type(inputs)=}"
    assert inputs.keys() == {'image'}
    assert isinstance(inputs['image'], torch.Tensor), f"{type(inputs['image'])=}"
    assert inputs['image'].ndim == 3 and inputs['image'].shape[0] == 3, f"{inputs['image'].shape=}"
    assert inputs['image'].dtype == torch.float32, f"{inputs['image'].dtype=}"
    assert -1 <= inputs['image'].min() <= inputs['image'].max() <= +1, f"{inputs['image'].min()=}, {inputs['image'].max()=}"


def validate_labels(labels: Dict[str, Any], dataset: MultiTaskFacialLandmarkDataset) -> None:
    assert isinstance(labels, dict), f"{type(labels)=}"
    assert labels.keys() == set(MultiTaskFacialLandmarkDataset.LABEL_NAMES)


def validate_meta_info(meta_info: Dict[str, Any], datapoint_idx: int, dataset: MultiTaskFacialLandmarkDataset) -> None:
    assert isinstance(meta_info, dict), f"{type(meta_info)=}"
    assert meta_info.keys() == {'idx', 'image_filepath', 'image_resolution'}
    assert meta_info['idx'] == datapoint_idx, f"meta_info['idx'] should match datapoint index: {meta_info['idx']=}, {datapoint_idx=}"

    image_filepath = meta_info['image_filepath']
    assert isinstance(image_filepath, str), f"{type(image_filepath)=}"
    assert os.path.isfile(os.path.join(dataset.data_root, image_filepath)), f"File does not exist: {os.path.join(dataset.data_root, image_filepath)}"

    image_resolution = meta_info['image_resolution']
    assert isinstance(image_resolution, tuple), f"{type(image_resolution)=}"
    assert len(image_resolution) == 2, f"{image_resolution=}"
    assert all(isinstance(x, int) for x in image_resolution), f"{image_resolution=}"
    assert all(x > 0 for x in image_resolution), f"{image_resolution=}"


def validate_datapoint(dataset: MultiTaskFacialLandmarkDataset, idx: int) -> None:
    datapoint = dataset[idx]
    assert isinstance(datapoint, dict), f"{type(datapoint)=}"
    assert datapoint.keys() == {'inputs', 'labels', 'meta_info'}
    validate_inputs(datapoint['inputs'])
    validate_labels(datapoint['labels'], dataset)
    validate_meta_info(datapoint['meta_info'], idx, dataset)
    # Additional validation for image resolution consistency
    assert datapoint['inputs']['image'].shape[-2:] == datapoint['meta_info']['image_resolution']


@pytest.fixture
def dataset(request, multi_task_facial_landmark_data_root):
    """Fixture for creating a MultiTaskFacialLandmarkDataset instance."""
    params = request.param
    return MultiTaskFacialLandmarkDataset(
        data_root=multi_task_facial_landmark_data_root,
        **params
    )


@pytest.mark.parametrize('dataset', [
    {
        'split': 'train',
    },
    {
        'split': 'train',
        'indices': [0, 2, 4, 6, 8],
    },
], indirect=True)
def test_multi_task_facial_landmark(dataset: MultiTaskFacialLandmarkDataset, max_samples, get_samples_to_test) -> None:
    assert isinstance(dataset, torch.utils.data.Dataset)
    assert len(dataset) > 0, "Dataset should not be empty"

    num_samples = get_samples_to_test(len(dataset), max_samples)
    indices = random.sample(range(len(dataset)), num_samples)
    with ThreadPoolExecutor() as executor:
        executor.map(lambda idx: validate_datapoint(dataset, idx), indices)
