from typing import Dict, Any
import pytest
import random
import torch
import os
from concurrent.futures import ThreadPoolExecutor
from data.datasets.multi_task_datasets.nyu_v2_dataset import NYUv2Dataset
from utils.builders.builder import build_from_config


def validate_inputs(inputs: Dict[str, Any]) -> None:
    assert isinstance(inputs, dict), f"{type(inputs)=}"
    assert inputs.keys() == {'image'}
    assert isinstance(inputs['image'], torch.Tensor), f"{type(inputs['image'])=}"
    assert inputs['image'].ndim == 3 and inputs['image'].shape[0] == 3, f"{inputs['image'].shape=}"
    assert inputs['image'].dtype == torch.float32, f"{inputs['image'].dtype=}"
    assert -1 <= inputs['image'].min() <= inputs['image'].max() <= +1, f"{inputs['image'].min()=}, {inputs['image'].max()=}"


def validate_labels(labels: Dict[str, Any], dataset: NYUv2Dataset, image_resolution: tuple) -> None:
    assert isinstance(labels, dict), f"{type(labels)=}"
    # Check that all label keys are valid NYUv2 labels
    assert set(labels.keys()).issubset(set(NYUv2Dataset.LABEL_NAMES)), f"Invalid label keys: {set(labels.keys()) - set(NYUv2Dataset.LABEL_NAMES)}"
    # Ensure at least one label is present
    assert len(labels) > 0, "At least one label must be present"

    # Conditionally validate each label if present
    if 'edge_detection' in labels:
        edge_detection = labels['edge_detection']
        assert isinstance(edge_detection, torch.Tensor), f"{type(edge_detection)=}"
        assert edge_detection.ndim == 3 and edge_detection.shape[0] == 1, f"{edge_detection.shape=}"
        assert edge_detection.dtype == torch.float32, f"{edge_detection.dtype=}"
        assert set(edge_detection.unique().tolist()) == set([0, 1]), f"{edge_detection.unique().tolist()=}"
        assert edge_detection.shape[-2:] == image_resolution, f"{edge_detection.shape=}, {image_resolution=}"

    if 'depth_estimation' in labels:
        depth_estimation = labels['depth_estimation']
        assert isinstance(depth_estimation, torch.Tensor), f"{type(depth_estimation)=}"
        assert depth_estimation.ndim == 2, f"{depth_estimation.shape=}"
        assert depth_estimation.dtype == torch.float32, f"{depth_estimation.dtype=}"
        assert depth_estimation.shape == image_resolution, f"{depth_estimation.shape=}, {image_resolution=}"

    if 'normal_estimation' in labels:
        normal_estimation = labels['normal_estimation']
        assert isinstance(normal_estimation, torch.Tensor), f"{type(normal_estimation)=}"
        assert normal_estimation.ndim == 3 and normal_estimation.shape[0] == 3, f"{normal_estimation.shape=}"
        assert normal_estimation.dtype == torch.float32, f"{normal_estimation.dtype=}"
        assert normal_estimation.shape[-2:] == image_resolution, f"{normal_estimation.shape=}, {image_resolution=}"

    if 'semantic_segmentation' in labels:
        semantic_segmentation = labels['semantic_segmentation']
        assert isinstance(semantic_segmentation, torch.Tensor), f"{type(semantic_segmentation)=}"
        assert semantic_segmentation.ndim == 2, f"{semantic_segmentation.shape=}"
        assert semantic_segmentation.dtype == torch.int64, f"{semantic_segmentation.dtype=}"
        assert semantic_segmentation.shape == image_resolution, f"{semantic_segmentation.shape=}, {image_resolution=}"
        assert set(semantic_segmentation.unique().tolist()).issubset(set(range(dataset.NUM_CLASSES)))


def validate_meta_info(meta_info: Dict[str, Any], datapoint_idx: int, dataset: NYUv2Dataset) -> None:
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


def validate_datapoint(dataset: NYUv2Dataset, idx: int) -> None:
    datapoint = dataset[idx]
    assert isinstance(datapoint, dict), f"{type(datapoint)=}"
    assert datapoint.keys() == {'inputs', 'labels', 'meta_info'}
    validate_inputs(datapoint['inputs'])
    validate_labels(datapoint['labels'], dataset, datapoint['meta_info']['image_resolution'])
    validate_meta_info(datapoint['meta_info'], idx, dataset)
    # Additional validation for image resolution consistency
    assert datapoint['inputs']['image'].shape[-2:] == datapoint['meta_info']['image_resolution']


@pytest.mark.parametrize('nyu_v2_dataset_config', [
    {
        'split': 'train',
    },
    {
        'split': 'train',
        'indices': [0, 2, 4, 6, 8],
    },
], indirect=True)
def test_nyu_v2(nyu_v2_dataset_config, max_samples, get_samples_to_test) -> None:
    dataset = build_from_config(nyu_v2_dataset_config)
    assert isinstance(dataset, torch.utils.data.Dataset)
    assert len(dataset) > 0, "Dataset should not be empty"

    num_samples = get_samples_to_test(len(dataset), max_samples)
    indices = random.sample(range(len(dataset)), num_samples)
    with ThreadPoolExecutor() as executor:
        executor.map(lambda idx: validate_datapoint(dataset, idx), indices)


@pytest.mark.parametrize('selected_labels,expected_keys', [
    (['depth_estimation'], ['depth_estimation']),
    (['normal_estimation'], ['normal_estimation']),
    (['semantic_segmentation'], ['semantic_segmentation']),
    (['edge_detection'], ['edge_detection']),
    (['depth_estimation', 'normal_estimation'], ['depth_estimation', 'normal_estimation']),
    (['semantic_segmentation', 'edge_detection'], ['semantic_segmentation', 'edge_detection']),
    (['depth_estimation', 'semantic_segmentation', 'edge_detection'], ['depth_estimation', 'semantic_segmentation', 'edge_detection']),
    (None, None),  # Default case - will be set to NYUv2Dataset.LABEL_NAMES
])
def test_nyu_v2_selective_loading(nyu_v2_base_config, selected_labels, expected_keys):
    """Test that NYUv2 dataset selective loading works correctly."""
    # Copy and modify the base config for selective loading
    import copy
    config = copy.deepcopy(nyu_v2_base_config)
    config['args']['labels'] = selected_labels
    
    dataset = build_from_config(config)
    
    # Check selected_labels attribute
    if selected_labels is None:
        assert dataset.selected_labels == NYUv2Dataset.LABEL_NAMES
        expected_keys = NYUv2Dataset.LABEL_NAMES  # Use the actual LABEL_NAMES for default case
    else:
        assert dataset.selected_labels == selected_labels
    
    # Load a datapoint and check only expected labels are present
    datapoint = dataset[0]
    
    assert isinstance(datapoint, dict)
    assert 'labels' in datapoint
    
    labels = datapoint['labels']
    assert isinstance(labels, dict)
    assert set(labels.keys()) == set(expected_keys)
    
    # Validate the labels using existing validation function
    validate_labels(labels, dataset, datapoint['meta_info']['image_resolution'])
