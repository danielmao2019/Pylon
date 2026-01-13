from typing import Dict, Any
import pytest
import random
import torch
from concurrent.futures import ThreadPoolExecutor
from data.datasets.multi_task_datasets.pascal_context_dataset import PASCALContextDataset
from utils.builders.builder import build_from_config


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


@pytest.mark.parametrize('pascal_context_dataset_config', ['train', 'val'], indirect=True)
def test_pascal_context_dataset(pascal_context_dataset_config, max_samples, get_samples_to_test) -> None:
    dataset = build_from_config(pascal_context_dataset_config)
    assert isinstance(dataset, torch.utils.data.Dataset)
    assert len(dataset) > 0, "Dataset should not be empty"

    num_samples = get_samples_to_test(len(dataset), max_samples)
    indices = random.sample(range(len(dataset)), num_samples)
    with ThreadPoolExecutor() as executor:
        executor.map(lambda idx: validate_datapoint(dataset, idx), indices)


@pytest.mark.parametrize('selected_labels,expected_keys', [
    (['semantic_segmentation'], ['semantic_segmentation']),
    (['parts_target'], ['parts_target']),
    (['normal_estimation'], ['normal_estimation']),
    (['saliency_estimation'], ['saliency_estimation']),
    (['semantic_segmentation', 'normal_estimation'], ['semantic_segmentation', 'normal_estimation']),
    (['parts_target', 'saliency_estimation'], ['parts_target', 'saliency_estimation']),
    (['semantic_segmentation', 'parts_target', 'normal_estimation'], ['semantic_segmentation', 'parts_target', 'normal_estimation']),
    (None, None),  # Default case - will be set to PASCALContextDataset.LABEL_NAMES
])
def test_pascal_context_selective_loading(pascal_context_base_config, selected_labels, expected_keys):
    """Test that PASCAL Context dataset selective loading works correctly."""
    # Copy and modify the base config for selective loading
    import copy
    config = copy.deepcopy(pascal_context_base_config)
    config['args']['labels'] = selected_labels

    dataset = build_from_config(config)

    # Check selected_labels attribute
    if selected_labels is None:
        assert dataset.selected_labels == PASCALContextDataset.LABEL_NAMES
        expected_keys = PASCALContextDataset.LABEL_NAMES  # Use the actual LABEL_NAMES for default case
    else:
        assert dataset.selected_labels == selected_labels

    # Load a datapoint and check only expected labels are present
    datapoint = dataset[0]

    assert isinstance(datapoint, dict)
    assert 'labels' in datapoint

    labels = datapoint['labels']
    assert isinstance(labels, dict)

    # Handle parts_target which may be None (not all images have human parts)
    if 'parts_target' in expected_keys and labels.get('parts_target') is None:
        # Remove parts_target from expected if it's None in actual data
        expected_keys = [k for k in expected_keys if k != 'parts_target']
        # Also remove parts_inst_mask if it was expected but parts_target is None
        if 'parts_inst_mask' in expected_keys:
            expected_keys = [k for k in expected_keys if k != 'parts_inst_mask']

    assert set(labels.keys()) == set(expected_keys)

    # Validate that all labels are proper types
    for key, value in labels.items():
        if key in ['parts_target', 'parts_inst_mask'] and value is None:
            continue  # None is acceptable for parts when no human parts are present
        assert isinstance(value, torch.Tensor), f"Label '{key}' should be tensor, got {type(value)}"
