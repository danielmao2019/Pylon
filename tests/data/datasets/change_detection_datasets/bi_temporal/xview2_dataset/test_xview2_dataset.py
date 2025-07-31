from typing import Any, Dict
import pytest
import torch
from concurrent.futures import ThreadPoolExecutor
from data.datasets.change_detection_datasets.bi_temporal.xview2_dataset import xView2Dataset


def validate_inputs(inputs: Dict[str, Any]) -> None:
    """Validate the inputs of a datapoint."""
    assert isinstance(inputs, dict), f"{type(inputs)=}"
    assert set(inputs.keys()) == set(xView2Dataset.INPUT_NAMES)
    img_1 = inputs['img_1']
    img_2 = inputs['img_2']
    assert type(img_1) == torch.Tensor
    assert img_1.ndim == 3 and img_1.shape[0] == 3, f"{img_1.shape=}"
    assert img_1.dtype == torch.float32
    assert 0 <= img_1.min() <= img_1.max() <= 1
    assert type(img_2) == torch.Tensor
    assert img_2.ndim == 3 and img_2.shape[0] == 3, f"{img_2.shape=}"
    assert img_2.dtype == torch.float32
    assert 0 <= img_2.min() <= img_2.max() <= 1
    assert img_1.shape == img_2.shape, f"{img_1.shape=}, {img_2.shape=}"


def validate_labels(labels: Dict[str, Any]) -> None:
    """Validate the labels of a datapoint."""
    assert isinstance(labels, dict), f"{type(labels)=}"
    assert set(labels.keys()) == set(xView2Dataset.LABEL_NAMES)
    lbl_1 = labels['lbl_1']
    lbl_2 = labels['lbl_2']
    assert type(lbl_1) == torch.Tensor and lbl_1.ndim == 2 and lbl_1.dtype == torch.int64
    assert type(lbl_2) == torch.Tensor and lbl_2.ndim == 2 and lbl_2.dtype == torch.int64
    assert set(torch.unique(lbl_1).tolist()).issubset(set([0, 1, 2, 3, 4])), f"{torch.unique(lbl_1)=}"
    assert set(torch.unique(lbl_2).tolist()).issubset(set([0, 1, 2, 3, 4])), f"{torch.unique(lbl_2)=}"


def validate_meta_info(meta_info: Dict[str, Any], datapoint_idx: int) -> None:
    """Validate the meta_info of a datapoint."""
    assert isinstance(meta_info, dict), f"{type(meta_info)=}"
    assert 'idx' in meta_info, f"meta_info should contain 'idx' key: {meta_info.keys()=}"
    assert meta_info['idx'] == datapoint_idx, f"meta_info['idx'] should match datapoint index: {meta_info['idx']=}, {datapoint_idx=}"


@pytest.mark.parametrize('dataset', ['train', 'test', 'hold'], indirect=True)
def test_xview2(dataset, max_samples, get_samples_to_test) -> None:
    assert isinstance(dataset, torch.utils.data.Dataset)
    assert len(dataset) > 0, "Dataset should not be empty"

    def validate_datapoint(idx: int) -> None:
        datapoint = dataset[idx]
        assert isinstance(datapoint, dict), f"{type(datapoint)=}"
        assert datapoint.keys() == {'inputs', 'labels', 'meta_info'}
        validate_inputs(datapoint['inputs'])
        validate_labels(datapoint['labels'])
        validate_meta_info(datapoint['meta_info'], idx)

    num_samples = get_samples_to_test(len(dataset), max_samples)
    indices = list(range(num_samples))
    with ThreadPoolExecutor() as executor:
        executor.map(validate_datapoint, indices)
