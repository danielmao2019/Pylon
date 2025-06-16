import pytest
import torch
from concurrent.futures import ThreadPoolExecutor
from data.datasets import ADE20KDataset


def validate_inputs(inputs: dict) -> None:
    assert isinstance(inputs, dict), f"{type(inputs)=}"
    assert inputs.keys() == {'image'}
    assert isinstance(inputs['image'], torch.Tensor), f"{type(inputs['image'])=}"
    assert inputs['image'].ndim == 3 and inputs['image'].shape[0] == 3, f"{inputs['image'].shape=}"
    assert inputs['image'].dtype == torch.float32, f"{inputs['image'].dtype=}"
    assert inputs['image'].min() >= 0.0 and inputs['image'].max() <= 1.0, f"{inputs['image'].min()=}, {inputs['image'].max()=}"


def validate_labels(labels: dict, image_resolution: tuple) -> None:
    assert isinstance(labels, dict), f"{type(labels)=}"
    assert labels.keys() == {'object_cls_mask', 'object_ins_mask', 'parts_cls_masks', 'parts_ins_masks', 'objects', 'parts', 'amodal_masks'}

    # Validate object_cls_mask
    assert isinstance(labels['object_cls_mask'], torch.Tensor), f"{type(labels['object_cls_mask'])=}"
    assert labels['object_cls_mask'].ndim == 2, f"{labels['object_cls_mask'].shape=}"
    assert labels['object_cls_mask'].dtype == torch.int64, f"{labels['object_cls_mask'].dtype=}"
    assert labels['object_cls_mask'].shape == image_resolution, f"{labels['object_cls_mask'].shape=}, {image_resolution=}"

    # Validate object_ins_mask
    assert isinstance(labels['object_ins_mask'], torch.Tensor), f"{type(labels['object_ins_mask'])=}"
    assert labels['object_ins_mask'].ndim == 2, f"{labels['object_ins_mask'].shape=}"
    assert labels['object_ins_mask'].dtype == torch.int64, f"{labels['object_ins_mask'].dtype=}"
    assert labels['object_ins_mask'].shape == image_resolution, f"{labels['object_ins_mask'].shape=}, {image_resolution=}"

    # Validate parts_cls_masks
    assert isinstance(labels['parts_cls_masks'], list), f"{type(labels['parts_cls_masks'])=}"
    assert all(isinstance(x, torch.Tensor) for x in labels['parts_cls_masks']), f"{labels['parts_cls_masks']=}"
    assert all(x.ndim == 2 for x in labels['parts_cls_masks']), f"{labels['parts_cls_masks']=}"
    assert all(x.dtype == torch.int64 for x in labels['parts_cls_masks']), f"{labels['parts_cls_masks']=}"
    assert all(x.shape == image_resolution for x in labels['parts_cls_masks']), f"{labels['parts_cls_masks']=}, {image_resolution=}"

    # Validate parts_ins_masks
    assert isinstance(labels['parts_ins_masks'], list), f"{type(labels['parts_ins_masks'])=}"
    assert all(isinstance(x, torch.Tensor) for x in labels['parts_ins_masks']), f"{labels['parts_ins_masks']=}"
    assert all(x.ndim == 2 for x in labels['parts_ins_masks']), f"{labels['parts_ins_masks']=}"
    assert all(x.dtype == torch.int64 for x in labels['parts_ins_masks']), f"{labels['parts_ins_masks']=}"
    assert all(x.shape == image_resolution for x in labels['parts_ins_masks']), f"{labels['parts_ins_masks']=}, {image_resolution=}"

    # Validate objects
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

    # Validate parts
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

    # Validate amodal_masks
    assert isinstance(labels['amodal_masks'], list), f"{type(labels['amodal_masks'])=}"
    assert all(isinstance(x, torch.Tensor) for x in labels['amodal_masks']), f"{labels['amodal_masks']=}"
    assert all(x.ndim == 2 for x in labels['amodal_masks']), f"{labels['amodal_masks']=}"
    assert all(x.dtype == torch.int64 for x in labels['amodal_masks']), f"{labels['amodal_masks']=}"
    assert all(x.shape == image_resolution for x in labels['amodal_masks']), f"{labels['amodal_masks']=}, {image_resolution=}"


def validate_meta_info(meta_info: dict) -> None:
    assert isinstance(meta_info, dict), f"{type(meta_info)=}"
    assert meta_info.keys() == {'image_filepath', 'object_mask_filepath', 'parts_masks_filepaths', 'attr_filepath', 'amodal_masks_filepaths', 'image_resolution'}
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


@pytest.mark.parametrize('split', ['training', 'validation'])
def test_ade_20k(split: str):
    dataset = ADE20KDataset(
        data_root='./data/datasets/soft_links/ADE20K',
        split=split,
    )
    assert dataset.split == split, f"{dataset.split=}, {split=}"

    def validate_datapoint(idx: int) -> None:
        dp = dataset[idx]
        assert isinstance(dp, dict), f"{type(dp)=}"
        assert dp.keys() == {'inputs', 'labels', 'meta_info'}
        validate_inputs(dp['inputs'])
        validate_labels(dp['labels'], dp['meta_info']['image_resolution'])
        validate_meta_info(dp['meta_info'])

    with ThreadPoolExecutor() as executor:
        executor.map(validate_datapoint, range(len(dataset)))
