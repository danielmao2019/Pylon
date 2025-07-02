from typing import Dict, Any
import pytest
import random
import torch
import os
from concurrent.futures import ThreadPoolExecutor
from data.datasets.multi_task_datasets.celeb_a_dataset import CelebADataset


def validate_inputs(inputs: Dict[str, Any], image_resolution: tuple) -> None:
    assert isinstance(inputs, dict), f"{type(inputs)=}"
    assert inputs.keys() == {'image'}
    image = inputs['image']
    assert isinstance(image, torch.Tensor), f"{type(image)=}"
    assert image.ndim == 3 and image.shape[0] == 3, f"{image.shape=}"
    assert image.dtype == torch.float32, f"{image.dtype=}"
    assert -1 <= image.min() <= image.max() <= +1, f"{image.min()=}, {image.max()=}"
    assert image.shape[-2:] == image_resolution, f"{image.shape[-2:]=}, {image_resolution=}"


def validate_labels(labels: Dict[str, Any], dataset: CelebADataset) -> None:
    assert isinstance(labels, dict), f"{type(labels)=}"
    assert labels.keys() == set(CelebADataset.LABEL_NAMES) - set(['landmarks'])


def validate_meta_info(meta_info: Dict[str, Any], datapoint_idx: int, dataset: CelebADataset) -> None:
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


@pytest.fixture
def dataset(request):
    """Fixture for creating a CelebADataset instance."""
    params = request.param
    return CelebADataset(
        data_root='./data/datasets/soft_links/celeb-a',
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
def test_celeb_a(dataset: CelebADataset, max_samples, get_samples_to_test) -> None:
    assert isinstance(dataset, torch.utils.data.Dataset)

    def validate_datapoint(idx: int) -> None:
        datapoint = dataset[idx]
        assert isinstance(datapoint, dict), f"{type(datapoint)=}"
        assert datapoint.keys() == {'inputs', 'labels', 'meta_info'}
        validate_inputs(datapoint['inputs'], datapoint['meta_info']['image_resolution'])
        validate_labels(datapoint['labels'], dataset)
        validate_meta_info(datapoint['meta_info'], idx, dataset)

    num_samples = get_samples_to_test(len(dataset), max_samples, default=3)
    indices = random.sample(range(len(dataset)), num_samples)
    with ThreadPoolExecutor() as executor:
        executor.map(validate_datapoint, indices)
