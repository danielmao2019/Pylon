from typing import Dict, Any
import pytest
import random
import torch
import os
from concurrent.futures import ThreadPoolExecutor
from data.datasets.multi_task_datasets.city_scapes_dataset import CityScapesDataset


def validate_inputs(inputs: Dict[str, Any], image_resolution: tuple) -> None:
    assert isinstance(inputs, dict), f"{type(inputs)=}"
    assert inputs.keys() == {'image'}
    image = inputs['image']
    assert isinstance(image, torch.Tensor), f"{type(image)=}"
    assert image.ndim == 3 and image.shape[0] == 3, f"{image.shape=}"
    assert image.dtype == torch.float32, f"{image.dtype=}"
    assert -1 <= image.min() <= image.max() <= +1, f"{image.min()=}, {image.max()=}"
    assert image.shape[-2:] == image_resolution, f"{image.shape[-2:]=}, {image_resolution=}"


def validate_labels(labels: Dict[str, Any], dataset: CityScapesDataset, image_resolution: tuple) -> None:
    assert isinstance(labels, dict), f"{type(labels)=}"
    assert labels.keys() == {'depth_estimation', 'semantic_segmentation', 'instance_segmentation'}

    depth_estimation = labels['depth_estimation']
    assert isinstance(depth_estimation, torch.Tensor), f"{type(depth_estimation)=}"
    assert depth_estimation.ndim == 2, f"{depth_estimation.shape=}"
    assert depth_estimation.dtype == torch.float32, f"{depth_estimation.dtype=}"
    assert depth_estimation.shape == image_resolution, f"{depth_estimation.shape=}, {image_resolution=}"

    semantic_segmentation = labels['semantic_segmentation']
    assert isinstance(semantic_segmentation, torch.Tensor), f"{type(semantic_segmentation)=}"
    assert semantic_segmentation.ndim == 2, f"{semantic_segmentation.shape=}"
    assert semantic_segmentation.dtype == torch.int64, f"{semantic_segmentation.dtype=}"
    assert semantic_segmentation.shape == image_resolution, f"{semantic_segmentation.shape=}, {image_resolution=}"
    assert set(semantic_segmentation.unique().tolist()).issubset(set(list(dataset.CLASS_MAP.values()) + [dataset.IGNORE_INDEX]))

    instance_segmentation = labels['instance_segmentation']
    assert isinstance(instance_segmentation, torch.Tensor), f"{type(instance_segmentation)=}"
    assert instance_segmentation.ndim == 3 and instance_segmentation.shape[0] == 2, f"{instance_segmentation.shape=}"
    assert instance_segmentation.dtype == torch.float32, f"{instance_segmentation.dtype=}"
    assert instance_segmentation.shape[-2:] == image_resolution, f"{instance_segmentation.shape=}, {image_resolution=}"


def validate_meta_info(meta_info: Dict[str, Any], datapoint_idx: int, dataset: CityScapesDataset) -> None:
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
    """Fixture for creating a CityScapesDataset instance."""
    params = request.param
    return CityScapesDataset(
        data_root='./data/datasets/soft_links/city-scapes',
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
def test_city_scapes(dataset: CityScapesDataset, max_samples, get_samples_to_test) -> None:
    assert isinstance(dataset, torch.utils.data.Dataset)

    def validate_datapoint(idx: int) -> None:
        datapoint = dataset[idx]
        assert isinstance(datapoint, dict), f"{type(datapoint)=}"
        assert datapoint.keys() == {'inputs', 'labels', 'meta_info'}
        validate_inputs(datapoint['inputs'], datapoint['meta_info']['image_resolution'])
        validate_labels(datapoint['labels'], dataset, datapoint['meta_info']['image_resolution'])
        validate_meta_info(datapoint['meta_info'], idx, dataset)

    num_samples = get_samples_to_test(len(dataset), max_samples, default=3)
    indices = random.sample(range(len(dataset)), num_samples)
    with ThreadPoolExecutor() as executor:
        executor.map(validate_datapoint, indices)
