from typing import Dict, Any
import pytest
import random
import torch
from concurrent.futures import ThreadPoolExecutor
from data.datasets.pcr_datasets.kitti_dataset import KITTIDataset


def validate_inputs(inputs: Dict[str, Any]) -> None:
    assert isinstance(inputs, dict), f"{type(inputs)=}"
    assert inputs.keys() == {'src_pc', 'tgt_pc'}, f"{inputs.keys()=}"

    for pc_name in ['src_pc', 'tgt_pc']:
        pc = inputs[pc_name]
        assert isinstance(pc, dict), f"{pc_name} is not a dict: {type(pc)=}"
        assert pc.keys() == {'pos', 'reflectance'}, f"{pc_name} keys incorrect: {pc.keys()=}"

        # Validate position tensor
        assert isinstance(pc['pos'], torch.Tensor), f"{pc_name}['pos'] is not torch.Tensor: {type(pc['pos'])=}"
        assert pc['pos'].ndim == 2, f"{pc_name}['pos'] should be 2-dimensional: {pc['pos'].shape=}"
        assert pc['pos'].shape[1] == 3, f"{pc_name}['pos'] should have 3 coordinates: {pc['pos'].shape=}"
        assert pc['pos'].dtype == torch.float32, f"{pc_name}['pos'] dtype incorrect: {pc['pos'].dtype=}"

        # Validate reflectance tensor
        assert isinstance(pc['reflectance'], torch.Tensor), f"{pc_name}['reflectance'] is not torch.Tensor: {type(pc['reflectance'])=}"
        assert pc['reflectance'].ndim == 2, f"{pc_name}['reflectance'] should be 2-dimensional: {pc['reflectance'].shape=}"
        assert pc['reflectance'].shape[1] == 1, f"{pc_name}['reflectance'] should have 1 feature: {pc['reflectance'].shape=}"
        assert pc['reflectance'].dtype == torch.float32, f"{pc_name}['reflectance'] dtype incorrect: {pc['reflectance'].dtype=}"

        # Check shapes match
        assert pc['pos'].shape[0] == pc['reflectance'].shape[0], \
            f"{pc_name} positions and reflectance should have same number of points: {pc['pos'].shape[0]=}, {pc['reflectance'].shape[0]=}"


def validate_labels(labels: Dict[str, Any]) -> None:
    assert isinstance(labels, dict), f"{type(labels)=}"
    assert labels.keys() == {'transform'}, f"{labels.keys()=}"
    assert isinstance(labels['transform'], torch.Tensor), f"transform is not torch.Tensor: {type(labels['transform'])=}"
    assert labels['transform'].shape == (4, 4), f"transform shape incorrect: {labels['transform'].shape=}"
    assert labels['transform'].dtype == torch.float32, f"transform dtype incorrect: {labels['transform'].dtype=}"


def validate_meta_info(meta_info: Dict[str, Any], datapoint_idx: int) -> None:
    assert isinstance(meta_info, dict), f"{type(meta_info)=}"
    assert meta_info.keys() == {'idx', 'seq', 't0', 't1'}, f"{meta_info.keys()=}"
    assert meta_info['idx'] == datapoint_idx, f"{meta_info['idx']=}, {datapoint_idx=}"
    assert isinstance(meta_info['seq'], str), f"seq is not str: {type(meta_info['seq'])=}"
    assert isinstance(meta_info['t0'], int), f"{type(meta_info['t0'])=}"
    assert isinstance(meta_info['t1'], int), f"{type(meta_info['t1'])=}"


@pytest.fixture
def dataset(request):
    """Fixture for creating a KITTIDataset instance."""
    split = request.param
    return KITTIDataset(
        data_root='./data/datasets/soft_links/KITTI',
        split=split,
    )


@pytest.mark.parametrize('dataset', ['train', 'val', 'test'], indirect=True)
def test_kitti_dataset(dataset, max_samples, get_samples_to_test):
    """Test the structure and content of dataset outputs."""

    def validate_datapoint(idx: int) -> None:
        datapoint = dataset[idx]
        assert isinstance(datapoint, dict), f"{type(datapoint)=}"
        assert datapoint.keys() == {'inputs', 'labels', 'meta_info'}
        validate_inputs(datapoint['inputs'])
        validate_labels(datapoint['labels'])
        validate_meta_info(datapoint['meta_info'], idx)

    # Use command line --samples if provided, otherwise test first 5 samples
    num_samples = get_samples_to_test(len(dataset), max_samples, default=5)
    indices = random.sample(range(len(dataset)), num_samples)
    with ThreadPoolExecutor() as executor:
        executor.map(validate_datapoint, indices)
