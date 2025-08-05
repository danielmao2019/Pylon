from typing import Tuple, Dict, Any
import pytest
import random
import torch
from concurrent.futures import ThreadPoolExecutor
from data.datasets import ADE20KDataset


def validate_inputs(inputs: Dict[str, Any]) -> None:
    assert isinstance(inputs, dict), f"{type(inputs)=}"
    assert inputs.keys() == {'image'}
    assert isinstance(inputs['image'], torch.Tensor), f"{type(inputs['image'])=}"
    assert inputs['image'].ndim == 3 and inputs['image'].shape[0] == 3, f"{inputs['image'].shape=}"
    assert inputs['image'].dtype == torch.float32, f"{inputs['image'].dtype=}"
    assert inputs['image'].min() >= 0.0 and inputs['image'].max() <= 1.0, f"{inputs['image'].min()=}, {inputs['image'].max()=}"


def validate_labels(labels: Dict[str, Any], image_resolution: Tuple[int, int]) -> None:
    assert isinstance(labels, dict), f"{type(labels)=}"
    # Check that all label keys are valid ADE20K labels
    valid_ade20k_labels = {'object_cls_mask', 'object_ins_mask', 'parts_cls_masks', 'parts_ins_masks', 'objects', 'parts', 'amodal_masks'}
    assert set(labels.keys()).issubset(valid_ade20k_labels), f"Invalid label keys: {set(labels.keys()) - valid_ade20k_labels}"
    # Ensure at least one label is present
    assert len(labels) > 0, "At least one label must be present"

    # Conditionally validate each label if present
    if 'object_cls_mask' in labels:
        assert isinstance(labels['object_cls_mask'], torch.Tensor), f"{type(labels['object_cls_mask'])=}"
        assert labels['object_cls_mask'].ndim == 2, f"{labels['object_cls_mask'].shape=}"
        assert labels['object_cls_mask'].dtype == torch.int64, f"{labels['object_cls_mask'].dtype=}"
        assert labels['object_cls_mask'].shape == image_resolution, f"{labels['object_cls_mask'].shape=}, {image_resolution=}"

    if 'object_ins_mask' in labels:
        assert isinstance(labels['object_ins_mask'], torch.Tensor), f"{type(labels['object_ins_mask'])=}"
        assert labels['object_ins_mask'].ndim == 2, f"{labels['object_ins_mask'].shape=}"
        assert labels['object_ins_mask'].dtype == torch.int64, f"{labels['object_ins_mask'].dtype=}"
        assert labels['object_ins_mask'].shape == image_resolution, f"{labels['object_ins_mask'].shape=}, {image_resolution=}"

    if 'parts_cls_masks' in labels:
        assert isinstance(labels['parts_cls_masks'], list), f"{type(labels['parts_cls_masks'])=}"
        assert all(isinstance(x, torch.Tensor) for x in labels['parts_cls_masks']), f"{labels['parts_cls_masks']=}"
        assert all(x.ndim == 2 for x in labels['parts_cls_masks']), f"{labels['parts_cls_masks']=}"
        assert all(x.dtype == torch.int64 for x in labels['parts_cls_masks']), f"{labels['parts_cls_masks']=}"
        assert all(x.shape == image_resolution for x in labels['parts_cls_masks']), f"{labels['parts_cls_masks']=}, {image_resolution=}"

    if 'parts_ins_masks' in labels:
        assert isinstance(labels['parts_ins_masks'], list), f"{type(labels['parts_ins_masks'])=}"
        assert all(isinstance(x, torch.Tensor) for x in labels['parts_ins_masks']), f"{labels['parts_ins_masks']=}"
        assert all(x.ndim == 2 for x in labels['parts_ins_masks']), f"{labels['parts_ins_masks']=}"
        assert all(x.dtype == torch.int64 for x in labels['parts_ins_masks']), f"{labels['parts_ins_masks']=}"
        assert all(x.shape == image_resolution for x in labels['parts_ins_masks']), f"{labels['parts_ins_masks']=}, {image_resolution=}"

    if 'objects' in labels:
        assert isinstance(labels['objects'], dict), f"{type(labels['objects'])=}"
        assert labels['objects'].keys() == {'instancendx', 'class', 'corrected_raw_name', 'iscrop', 'listattributes', 'polygon'}
        assert isinstance(labels['objects']['instancendx'], torch.Tensor), f"{type(labels['objects']['instancendx'])=}"
        assert isinstance(labels['objects']['class'], list), f"{type(labels['objects']['class'])=}"
        assert isinstance(labels['objects']['corrected_raw_name'], list), f"{type(labels['objects']['corrected_raw_name'])=}"
        assert isinstance(labels['objects']['iscrop'], list), f"{type(labels['objects']['iscrop'])=}"
        assert isinstance(labels['objects']['listattributes'], list), f"{type(labels['objects']['listattributes'])=}"
        assert isinstance(labels['objects']['polygon'], list), f"{type(labels['objects']['polygon'])=}"
        assert all(isinstance(x, dict) for x in labels['objects']['polygon']), f"{labels['objects']['polygon']=}"
        assert all(isinstance(x['x'], torch.Tensor) for x in labels['objects']['polygon']), f"{labels['objects']['polygon']=}"
        assert all(isinstance(x['y'], torch.Tensor) for x in labels['objects']['polygon']), f"{labels['objects']['polygon']=}"

    if 'parts' in labels:
        assert isinstance(labels['parts'], dict), f"{type(labels['parts'])=}"
        assert labels['parts'].keys() == {'instancendx', 'class', 'corrected_raw_name', 'iscrop', 'listattributes', 'polygon'}
        assert isinstance(labels['parts']['instancendx'], torch.Tensor), f"{type(labels['parts']['instancendx'])=}"
        assert isinstance(labels['parts']['class'], list), f"{type(labels['parts']['class'])=}"
        assert isinstance(labels['parts']['corrected_raw_name'], list), f"{type(labels['parts']['corrected_raw_name'])=}"
        assert isinstance(labels['parts']['iscrop'], list), f"{type(labels['parts']['iscrop'])=}"
        assert isinstance(labels['parts']['listattributes'], list), f"{type(labels['parts']['listattributes'])=}"
        assert isinstance(labels['parts']['polygon'], list), f"{type(labels['parts']['polygon'])=}"
        assert all(isinstance(x, dict) for x in labels['parts']['polygon']), f"{labels['parts']['polygon']=}"
        assert all(isinstance(x['x'], torch.Tensor) for x in labels['parts']['polygon']), f"{labels['parts']['polygon']=}"
        assert all(isinstance(x['y'], torch.Tensor) for x in labels['parts']['polygon']), f"{labels['parts']['polygon']=}"

    if 'amodal_masks' in labels:
        assert isinstance(labels['amodal_masks'], list), f"{type(labels['amodal_masks'])=}"
        assert all(isinstance(x, torch.Tensor) for x in labels['amodal_masks']), f"{labels['amodal_masks']=}"
        assert all(x.ndim == 2 for x in labels['amodal_masks']), f"{labels['amodal_masks']=}"
        assert all(x.dtype == torch.int64 for x in labels['amodal_masks']), f"{labels['amodal_masks']=}"
        assert all(x.shape == image_resolution for x in labels['amodal_masks']), f"{labels['amodal_masks']=}, {image_resolution=}"


def validate_meta_info(meta_info: Dict[str, Any], datapoint_idx: int) -> None:
    assert isinstance(meta_info, dict), f"{type(meta_info)=}"
    assert meta_info.keys() == {'idx', 'image_filepath', 'object_mask_filepath', 'parts_masks_filepaths', 'attr_filepath', 'amodal_masks_filepaths', 'image_resolution'}
    assert meta_info['idx'] == datapoint_idx, f"meta_info['idx'] should match datapoint index: {meta_info['idx']=}, {datapoint_idx=}"
    assert isinstance(meta_info['image_filepath'], str), f"{meta_info['image_filepath']=}"
    assert isinstance(meta_info['object_mask_filepath'], str), f"{meta_info['object_mask_filepath']=}"
    assert isinstance(meta_info['parts_masks_filepaths'], list), f"{type(meta_info['parts_masks_filepaths'])=}"
    assert all(isinstance(x, str) for x in meta_info['parts_masks_filepaths']), f"{meta_info['parts_masks_filepaths']=}"
    assert isinstance(meta_info['attr_filepath'], str), f"{meta_info['attr_filepath']=}"
    assert isinstance(meta_info['amodal_masks_filepaths'], list), f"{type(meta_info['amodal_masks_filepaths'])=}"
    assert all(isinstance(x, str) for x in meta_info['amodal_masks_filepaths']), f"{meta_info['amodal_masks_filepaths']=}"
    assert isinstance(meta_info['image_resolution'], tuple), f"{type(meta_info['image_resolution'])=}"
    assert len(meta_info['image_resolution']) == 2, f"{meta_info['image_resolution']=}"
    assert all(isinstance(x, int) for x in meta_info['image_resolution']), f"{meta_info['image_resolution']=}"
    assert all(x > 0 for x in meta_info['image_resolution']), f"{meta_info['image_resolution']=}"


@pytest.fixture
def dataset(request, ade20k_data_root):
    """Fixture for creating an ADE20KDataset instance."""
    split = request.param
    return ADE20KDataset(
        data_root=ade20k_data_root,
        split=split,
    )


@pytest.mark.parametrize('dataset', ['training', 'validation'], indirect=True)
def test_ade_20k(dataset, max_samples, get_samples_to_test):
    assert isinstance(dataset, torch.utils.data.Dataset)
    assert len(dataset) > 0, "Dataset should not be empty"

    def validate_datapoint(idx: int) -> None:
        datapoint = dataset[idx]
        assert isinstance(datapoint, dict), f"{type(datapoint)=}"
        assert datapoint.keys() == {'inputs', 'labels', 'meta_info'}
        validate_inputs(datapoint['inputs'])
        validate_labels(datapoint['labels'], datapoint['meta_info']['image_resolution'])
        validate_meta_info(datapoint['meta_info'], idx)

    num_samples = get_samples_to_test(len(dataset), max_samples)
    indices = random.sample(range(len(dataset)), num_samples)
    with ThreadPoolExecutor() as executor:
        executor.map(validate_datapoint, indices)


@pytest.mark.parametrize('selected_labels,expected_keys', [
    (['object_cls_mask'], ['object_cls_mask']),
    (['object_ins_mask'], ['object_ins_mask']),
    (['objects'], ['objects']),
    (['parts'], ['parts']),
    (['object_cls_mask', 'object_ins_mask'], ['object_cls_mask', 'object_ins_mask']),
    (['objects', 'parts'], ['objects', 'parts']),
    (['object_cls_mask', 'objects'], ['object_cls_mask', 'objects']),
    (None, ['object_cls_mask', 'object_ins_mask', 'parts_cls_masks', 'parts_ins_masks', 'objects', 'parts', 'amodal_masks']),  # Default case
])
def test_ade_20k_selective_loading(ade20k_data_root, selected_labels, expected_keys):
    """Test that ADE20K dataset selective loading works correctly."""
    dataset = ADE20KDataset(
        data_root=ade20k_data_root,
        split='training',
        labels=selected_labels
    )
    
    # Check selected_labels attribute
    if selected_labels is None:
        assert dataset.selected_labels == dataset.LABEL_NAMES
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
    validate_labels(labels, datapoint['meta_info']['image_resolution'])
