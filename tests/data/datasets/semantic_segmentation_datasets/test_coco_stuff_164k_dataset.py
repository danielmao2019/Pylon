from typing import Tuple, Dict, Any
import pytest
import random
import torch
from concurrent.futures import ThreadPoolExecutor
from data.datasets import COCOStuff164KDataset


def validate_inputs(inputs: Dict[str, Any]) -> None:
    assert isinstance(inputs, dict), f"{type(inputs)=}"
    assert inputs.keys() == {'image'}
    assert isinstance(inputs['image'], torch.Tensor), f"{type(inputs['image'])=}"
    assert inputs['image'].ndim == 3 and inputs['image'].shape[0] == 3, f"{inputs['image'].shape=}"
    assert inputs['image'].dtype == torch.float32, f"{inputs['image'].dtype=}"
    assert inputs['image'].min() >= 0.0 and inputs['image'].max() <= 1.0, f"{inputs['image'].min()=}, {inputs['image'].max()=}"


def validate_labels(labels: Dict[str, Any], num_classes: int, image_resolution: Tuple[int, int]) -> None:
    assert isinstance(labels, dict), f"{type(labels)=}"
    assert labels.keys() == {'label'}
    assert isinstance(labels['label'], torch.Tensor), f"{type(labels['label'])=}"
    assert labels['label'].ndim == 2, f"{labels['label'].shape=}"
    assert labels['label'].dtype == torch.int64, f"{labels['label'].dtype=}"
    assert set(labels['label'].unique().tolist()).issubset(set(range(num_classes)) | {255}), f"{sorted(set(labels['label'].unique().tolist()))=}"
    assert labels['label'].shape == image_resolution, f"{labels['label'].shape=}, {image_resolution=}"


def validate_meta_info(meta_info: Dict[str, Any], datapoint_idx: int) -> None:
    assert isinstance(meta_info, dict), f"{type(meta_info)=}"
    assert meta_info.keys() == {'idx', 'image_filepath', 'label_filepath', 'image_resolution'}
    assert meta_info['idx'] == datapoint_idx, f"{meta_info['idx']=}, {datapoint_idx=}"


@pytest.mark.parametrize('split', ['train2017', 'val2017'])
@pytest.mark.parametrize('semantic_granularity', ['fine', 'coarse'])
def test_coco_stuff_164k(split: str, semantic_granularity: str, max_samples, get_samples_to_test):
    dataset = COCOStuff164KDataset(
        data_root='./data/datasets/soft_links/COCOStuff164K',
        split=split,
        semantic_granularity=semantic_granularity,
    )
    assert dataset.semantic_granularity == semantic_granularity, f"{dataset.semantic_granularity=}, {semantic_granularity=}"
    assert dataset.NUM_CLASSES == 182 if semantic_granularity == 'fine' else 27, f"{dataset.NUM_CLASSES=}, {semantic_granularity=}"

    def validate_datapoint(idx: int) -> None:
        datapoint = dataset[idx]
        assert isinstance(datapoint, dict), f"{type(datapoint)=}"
        assert datapoint.keys() == {'inputs', 'labels', 'meta_info'}
        validate_inputs(datapoint['inputs'])
        validate_labels(datapoint['labels'], dataset.NUM_CLASSES, datapoint['inputs']['image'].shape[-2:])
        validate_meta_info(datapoint['meta_info'], idx)

    # Use command line --samples if provided, otherwise default to 1000
    num_samples = get_samples_to_test(len(dataset), max_samples, default=1000)
    indices = random.sample(range(len(dataset)), num_samples)
    with ThreadPoolExecutor() as executor:
        executor.map(validate_datapoint, indices)
