from typing import Dict, Any
import pytest
import random
import torch
from concurrent.futures import ThreadPoolExecutor
from data.datasets.pcr_datasets.real_pcr_dataset import RealPCRDataset


def validate_inputs(inputs: Dict[str, Any]) -> None:
    assert isinstance(inputs, dict), f"{type(inputs)=}"
    assert inputs.keys() == {'src_pc', 'tgt_pc'}, f"{inputs.keys()=}"
    
    for pc_name in ['src_pc', 'tgt_pc']:
        pc = inputs[pc_name]
        assert isinstance(pc, dict), f"{pc_name} should be a dictionary: {type(pc)=}"
        assert pc.keys() == {'pos', 'feat'}, f"{pc_name} should contain exactly pos and feat: {pc.keys()=}"

        # Check shapes and types
        assert isinstance(pc['pos'], torch.Tensor), f"{pc_name}['pos'] should be a torch.Tensor: {type(pc['pos'])=}"
        assert isinstance(pc['feat'], torch.Tensor), f"{pc_name}['feat'] should be a torch.Tensor: {type(pc['feat'])=}"
        assert pc['pos'].dim() == 2, f"{pc_name}['pos'] should be 2-dimensional: {pc['pos'].shape=}"
        assert pc['pos'].size(1) == 3, f"{pc_name}['pos'] should have 3 coordinates: {pc['pos'].shape=}"
        assert pc['feat'].dim() == 2, f"{pc_name}['feat'] should be 2-dimensional: {pc['feat'].shape=}"
        assert pc['feat'].size(1) == 1, f"{pc_name}['feat'] should have 1 feature: {pc['feat'].shape=}"
        assert pc['pos'].dtype == torch.float32, f"{pc_name}['pos'] dtype incorrect: {pc['pos'].dtype=}"
        assert pc['feat'].dtype == torch.float32, f"{pc_name}['feat'] dtype incorrect: {pc['feat'].dtype=}"

        # Check for NaN values
        assert not torch.isnan(pc['pos']).any(), f"{pc_name}['pos'] contains NaN values"
        assert not torch.isnan(pc['feat']).any(), f"{pc_name}['feat'] contains NaN values"
        
        # Check that positions and features have same number of points
        assert pc['pos'].shape[0] == pc['feat'].shape[0], \
            f"{pc_name} positions and features should have same number of points: {pc['pos'].shape[0]=}, {pc['feat'].shape[0]=}"


def validate_labels(labels: Dict[str, Any]) -> None:
    assert isinstance(labels, dict), f"{type(labels)=}"
    assert labels.keys() == {'transform'}, f"{labels.keys()=}"
    assert isinstance(labels['transform'], torch.Tensor), f"transform should be a torch.Tensor: {type(labels['transform'])=}"
    assert labels['transform'].shape == (4, 4), f"transform should be a 4x4 matrix: {labels['transform'].shape=}"
    assert labels['transform'].dtype == torch.float32, f"transform dtype incorrect: {labels['transform'].dtype=}"
    assert not torch.isnan(labels['transform']).any(), "transform contains NaN values"

    # Validate transformation matrix
    transform = labels['transform']
    R = transform[:3, :3]
    t = transform[:3, 3]

    # Check rotation matrix properties
    assert torch.allclose(R @ R.T, torch.eye(3, device=R.device), atol=1e-6), \
        "Invalid rotation matrix: not orthogonal"
    assert torch.abs(torch.det(R) - 1.0) < 1e-6, \
        "Invalid rotation matrix: determinant not 1"


def validate_meta_info(meta_info: Dict[str, Any], datapoint_idx: int) -> None:
    assert isinstance(meta_info, dict), f"{type(meta_info)=}"
    assert meta_info.keys() >= {'idx', 'src_path', 'tgt_path'}, \
        f"meta_info missing required keys: {meta_info.keys()=}"
    assert meta_info['idx'] == datapoint_idx, f"{meta_info['idx']=}, {datapoint_idx=}"


@pytest.fixture
def dataset(request):
    """Fixture for creating a RealPCRDataset instance."""
    dataset_params = request.param
    return RealPCRDataset(**dataset_params)


@pytest.mark.parametrize('dataset', [
    {
        'data_root': './data/datasets/soft_links/ivision-pcr-data',
        'cache_dirname': 'real_pcr_cache',
        'gt_transforms_filepath': './data/datasets/soft_links/ivision-pcr-data/transforms.json',
        'split': 'train',
        'voxel_size': 10.0,
        'min_points': 256,
        'max_points': 8192,
    },
    {
        'data_root': './data/datasets/soft_links/ivision-pcr-data',
        'cache_dirname': 'real_pcr_cache',
        'gt_transforms_filepath': './data/datasets/soft_links/ivision-pcr-data/transforms.json',
        'split': 'val',
        'voxel_size': 10.0,
        'min_points': 256,
        'max_points': 8192,
    },
], indirect=True)
def test_real_pcr_dataset(dataset, max_samples, get_samples_to_test):
    """Test basic functionality of RealPCRDataset."""

    # Basic dataset checks
    assert len(dataset) > 0, "Dataset should not be empty"
    assert hasattr(dataset, 'annotations'), "Dataset should have annotations"

    def validate_datapoint(idx: int) -> None:
        datapoint = dataset[idx]
        assert isinstance(datapoint, dict), f"{type(datapoint)=}"
        assert datapoint.keys() == {'inputs', 'labels', 'meta_info'}
        validate_inputs(datapoint['inputs'])
        validate_labels(datapoint['labels'])
        validate_meta_info(datapoint['meta_info'], idx)

    # Use command line --samples if provided, otherwise test all samples
    num_samples = get_samples_to_test(len(dataset), max_samples)
    indices = random.sample(range(len(dataset)), num_samples)
    with ThreadPoolExecutor() as executor:
        executor.map(validate_datapoint, indices)
