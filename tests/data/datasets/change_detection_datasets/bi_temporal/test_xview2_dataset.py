from typing import Any, Dict
import pytest
import random
import torch
from concurrent.futures import ThreadPoolExecutor
from data.datasets.change_detection_datasets.bi_temporal.xview2_dataset import xView2Dataset


def validate_inputs(inputs: Dict[str, Any], dataset: xView2Dataset) -> None:
    """Validate the inputs of a datapoint."""
    assert set(inputs.keys()) == set(xView2Dataset.INPUT_NAMES)
    assert inputs['img_1'].ndim == 3 and inputs['img_1'].shape[0] == 3, f"{inputs['img_1'].shape=}"
    assert inputs['img_2'].ndim == 3 and inputs['img_2'].shape[0] == 3, f"{inputs['img_2'].shape=}"
    assert 0 <= inputs['img_1'].min() <= inputs['img_1'].max() <= 1
    assert 0 <= inputs['img_2'].min() <= inputs['img_2'].max() <= 1


def validate_labels(labels: Dict[str, Any], dataset: xView2Dataset) -> None:
    """Validate the labels of a datapoint."""
    assert set(labels.keys()) == set(xView2Dataset.LABEL_NAMES)
    assert labels['lbl_1'].ndim == 2, f"{labels['lbl_1'].shape=}"
    assert labels['lbl_2'].ndim == 2, f"{labels['lbl_2'].shape=}"
    assert set(torch.unique(labels['lbl_1']).tolist()).issubset(set([0, 1, 2, 3, 4])), f"{torch.unique(labels['lbl_1'])=}"
    assert set(torch.unique(labels['lbl_2']).tolist()).issubset(set([0, 1, 2, 3, 4])), f"{torch.unique(labels['lbl_2'])=}"


def validate_meta_info(meta_info: Dict[str, Any], datapoint_idx: int) -> None:
    """Validate the meta_info of a datapoint."""
    assert 'idx' in meta_info, f"meta_info should contain 'idx' key: {meta_info.keys()=}"
    assert meta_info['idx'] == datapoint_idx, f"meta_info['idx'] should match datapoint index: {meta_info['idx']=}, {datapoint_idx=}"


@pytest.mark.parametrize('split', ['train', 'test', 'hold'])
def test_xview2(split: str, max_samples) -> None:
    dataset = xView2Dataset(data_root="./data/datasets/soft_links/xView2", split=split)
    assert isinstance(dataset, torch.utils.data.Dataset)

    def validate_datapoint(idx: int) -> None:
        datapoint = dataset[idx]
        validate_inputs(datapoint['inputs'], dataset)
        validate_labels(datapoint['labels'], dataset)
        validate_meta_info(datapoint['meta_info'], idx)

    # Use command line --samples if provided, otherwise default to 100
    num_samples = min(len(dataset), max_samples if max_samples is not None else 100)
    indices = random.sample(range(len(dataset)), num_samples)
    with ThreadPoolExecutor() as executor:
        executor.map(validate_datapoint, indices)
