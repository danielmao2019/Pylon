from typing import Dict, Any
import pytest
import random
import os
import torch
from concurrent.futures import ThreadPoolExecutor
from data.datasets.change_detection_datasets.bi_temporal.slpccd_dataset import SLPCCDDataset
from tests.data.datasets.conftest import get_samples_to_test


def validate_inputs(inputs: Dict[str, Any]) -> None:
    assert isinstance(inputs, dict), f"{type(inputs)=}"
    assert set(inputs.keys()) == set(SLPCCDDataset.INPUT_NAMES)

    # Check point cloud data for both time points
    for pc_key in ['pc_1', 'pc_2']:
        assert pc_key in inputs
        assert 'xyz' in inputs[pc_key]
        assert isinstance(inputs[pc_key]['xyz'], torch.Tensor)
        assert inputs[pc_key]['xyz'].shape[1] == 3  # xyz coordinates


def validate_labels(labels: Dict[str, Any]) -> None:
    assert isinstance(labels, dict), f"{type(labels)=}"
    assert set(labels.keys()) == set(SLPCCDDataset.LABEL_NAMES)
    assert isinstance(labels['change_map'], torch.Tensor)


def validate_meta_info(meta_info: Dict[str, Any], datapoint_idx: int) -> None:
    assert isinstance(meta_info, dict), f"{type(meta_info)=}"
    assert 'idx' in meta_info, f"meta_info should contain 'idx' key: {meta_info.keys()=}"
    assert meta_info['idx'] == datapoint_idx, f"meta_info['idx'] should match datapoint index: {meta_info['idx']=}, {datapoint_idx=}"
    assert 'pc_1_filepath' in meta_info
    assert 'pc_2_filepath' in meta_info


@pytest.mark.parametrize('split', ['train', 'val', 'test'])
def test_load_real_dataset(split: str, max_samples) -> None:
    """Test loading the actual SLPCCD dataset."""
    dataset = SLPCCDDataset(
        data_root="./data/datasets/soft_links/SLPCCD",
        split=split,
        num_points=256,  # Use fewer points for faster testing
        use_hierarchy=False,  # Disable hierarchy for faster testing
        random_subsample=True
    )
    assert isinstance(dataset, torch.utils.data.Dataset)
    # Verify dataset has expected number of samples
    assert len(dataset) > 0, f"No data found in {split} split"

    def validate_datapoint(idx: int) -> None:
        datapoint = dataset[idx]
        assert isinstance(datapoint, dict), f"{type(datapoint)=}"
        assert datapoint.keys() == {'inputs', 'labels', 'meta_info'}
        validate_inputs(datapoint['inputs'])
        validate_labels(datapoint['labels'])
        validate_meta_info(datapoint['meta_info'], idx)

    num_samples = get_samples_to_test(len(dataset), max_samples, default=len(dataset))
    indices = list(range(num_samples))
    with ThreadPoolExecutor() as executor:
        executor.map(validate_datapoint, indices)
