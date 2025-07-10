from typing import Dict, Any
import pytest
import random
import torch
from concurrent.futures import ThreadPoolExecutor
from data.datasets.pcr_datasets.threedmatch_dataset import ThreeDMatchDataset


def validate_inputs(inputs: Dict[str, Any]) -> None:
    assert isinstance(inputs, dict), f"{type(inputs)=}"
    assert inputs.keys() == {'src_pc', 'tgt_pc', 'correspondences'}, f"{inputs.keys()=}"
    
    # Validate source point cloud
    src_pc = inputs['src_pc']
    assert isinstance(src_pc, dict), f"src_pc is not a dict: {type(src_pc)=}"
    assert src_pc.keys() == {'pos', 'feat'}, f"src_pc keys incorrect: {src_pc.keys()=}"
    
    assert isinstance(src_pc['pos'], torch.Tensor), f"src_pc['pos'] is not torch.Tensor: {type(src_pc['pos'])=}"
    assert src_pc['pos'].ndim == 2, f"src_pc['pos'] should be 2-dimensional: {src_pc['pos'].shape=}"
    assert src_pc['pos'].shape[1] == 3, f"src_pc['pos'] should have 3 coordinates: {src_pc['pos'].shape=}"
    assert src_pc['pos'].dtype == torch.float32, f"src_pc['pos'] dtype incorrect: {src_pc['pos'].dtype=}"
    
    assert isinstance(src_pc['feat'], torch.Tensor), f"src_pc['feat'] is not torch.Tensor: {type(src_pc['feat'])=}"
    assert src_pc['feat'].ndim == 2, f"src_pc['feat'] should be 2-dimensional: {src_pc['feat'].shape=}"
    assert src_pc['feat'].shape[1] == 1, f"src_pc['feat'] should have 1 feature: {src_pc['feat'].shape=}"
    assert src_pc['feat'].dtype == torch.float32, f"src_pc['feat'] dtype incorrect: {src_pc['feat'].dtype=}"
    assert src_pc['pos'].shape[0] == src_pc['feat'].shape[0], \
        f"src_pc positions and features should have same number of points: {src_pc['pos'].shape[0]=}, {src_pc['feat'].shape[0]=}"
    
    # Validate target point cloud
    tgt_pc = inputs['tgt_pc']
    assert isinstance(tgt_pc, dict), f"tgt_pc is not a dict: {type(tgt_pc)=}"
    assert tgt_pc.keys() == {'pos', 'feat'}, f"tgt_pc keys incorrect: {tgt_pc.keys()=}"
    
    assert isinstance(tgt_pc['pos'], torch.Tensor), f"tgt_pc['pos'] is not torch.Tensor: {type(tgt_pc['pos'])=}"
    assert tgt_pc['pos'].ndim == 2, f"tgt_pc['pos'] should be 2-dimensional: {tgt_pc['pos'].shape=}"
    assert tgt_pc['pos'].shape[1] == 3, f"tgt_pc['pos'] should have 3 coordinates: {tgt_pc['pos'].shape=}"
    assert tgt_pc['pos'].dtype == torch.float32, f"tgt_pc['pos'] dtype incorrect: {tgt_pc['pos'].dtype=}"
    
    assert isinstance(tgt_pc['feat'], torch.Tensor), f"tgt_pc['feat'] is not torch.Tensor: {type(tgt_pc['feat'])=}"
    assert tgt_pc['feat'].ndim == 2, f"tgt_pc['feat'] should be 2-dimensional: {tgt_pc['feat'].shape=}"
    assert tgt_pc['feat'].shape[1] == 1, f"tgt_pc['feat'] should have 1 feature: {tgt_pc['feat'].shape=}"
    assert tgt_pc['feat'].dtype == torch.float32, f"tgt_pc['feat'] dtype incorrect: {tgt_pc['feat'].dtype=}"
    assert tgt_pc['pos'].shape[0] == tgt_pc['feat'].shape[0], \
        f"tgt_pc positions and features should have same number of points: {tgt_pc['pos'].shape[0]=}, {tgt_pc['feat'].shape[0]=}"
    
    # Validate correspondences
    correspondences = inputs['correspondences']
    assert isinstance(correspondences, torch.Tensor), f"correspondences is not torch.Tensor: {type(correspondences)=}"
    assert correspondences.ndim == 2, f"correspondences should be 2-dimensional: {correspondences.shape=}"
    assert correspondences.shape[1] == 2, f"correspondences should have 2 columns (src, tgt indices): {correspondences.shape=}"
    assert correspondences.dtype == torch.int64, f"correspondences dtype incorrect: {correspondences.dtype=}"
    
    # Check correspondence indices are valid
    if correspondences.shape[0] > 0:
        assert torch.all(correspondences[:, 0] >= 0) and torch.all(correspondences[:, 0] < src_pc['pos'].shape[0]), \
            "Invalid source indices in correspondences"
        assert torch.all(correspondences[:, 1] >= 0) and torch.all(correspondences[:, 1] < tgt_pc['pos'].shape[0]), \
            "Invalid target indices in correspondences"


def validate_labels(labels: Dict[str, Any]) -> None:
    assert isinstance(labels, dict), f"{type(labels)=}"
    assert labels.keys() == {'transform'}, f"{labels.keys()=}"
    assert isinstance(labels['transform'], torch.Tensor), f"transform is not torch.Tensor: {type(labels['transform'])=}"
    assert labels['transform'].shape == (4, 4), f"transform shape incorrect: {labels['transform'].shape=}"
    assert labels['transform'].dtype == torch.float32, f"transform dtype incorrect: {labels['transform'].dtype=}"
    
    # Check transform is valid
    expected_last_row = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=labels['transform'].device)
    assert torch.allclose(labels['transform'][3, :], expected_last_row), \
        "Transform last row should be [0, 0, 0, 1]"
    
    # Check rotation is orthogonal
    rotation = labels['transform'][:3, :3]
    rotation_transpose = rotation.transpose(0, 1)
    identity = torch.eye(3, dtype=torch.float32, device=rotation.device)
    assert torch.allclose(rotation @ rotation_transpose, identity, atol=1e-5), \
        "Rotation matrix is not orthogonal"


def validate_meta_info(meta_info: Dict[str, Any], datapoint_idx: int) -> None:
    assert isinstance(meta_info, dict), f"{type(meta_info)=}"
    assert meta_info.keys() == {'idx', 'src_path', 'tgt_path', 'scene_name', 'overlap', 'src_frame', 'tgt_frame'}, \
        f"{meta_info.keys()=}"
    assert meta_info['idx'] == datapoint_idx, f"{meta_info['idx']=}, {datapoint_idx=}"
    assert isinstance(meta_info['src_path'], str), f"src_path is not str: {type(meta_info['src_path'])=}"
    assert isinstance(meta_info['tgt_path'], str), f"tgt_path is not str: {type(meta_info['tgt_path'])=}"
    assert isinstance(meta_info['scene_name'], str), f"scene_name is not str: {type(meta_info['scene_name'])=}"
    assert isinstance(meta_info['overlap'], float), f"overlap is not float: {type(meta_info['overlap'])=}"
    assert isinstance(meta_info['src_frame'], int), f"src_frame is not int: {type(meta_info['src_frame'])=}"
    assert isinstance(meta_info['tgt_frame'], int), f"tgt_frame is not int: {type(meta_info['tgt_frame'])=}"
    assert 0.0 <= meta_info['overlap'] <= 1.0, f"overlap should be between 0 and 1: {meta_info['overlap']=}"


@pytest.fixture
def dataset(request):
    """Fixture for creating a ThreeDMatchDataset instance."""
    split = request.param
    return ThreeDMatchDataset(
        data_root='./data/datasets/soft_links/threedmatch',
        split=split,
        num_points=5000,
        matching_radius=0.1,
        overlap_threshold=0.3,
        benchmark_mode='3DMatch',
    )


@pytest.mark.parametrize('dataset', ['train', 'val', 'test'], indirect=True)
def test_threedmatch_dataset(dataset, max_samples, get_samples_to_test):
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


def test_threedmatch_dataset_3dlomatch():
    """Test 3DLoMatch benchmark mode."""
    dataset = ThreeDMatchDataset(
        data_root='./data/datasets/soft_links/threedmatch',
        split='test',
        num_points=5000,
        matching_radius=0.1,
        overlap_threshold=0.1,  # Lower overlap threshold for 3DLoMatch
        benchmark_mode='3DLoMatch',
    )
    
    # Test a single sample to ensure it loads correctly
    if len(dataset) > 0:
        datapoint = dataset[0]
        validate_inputs(datapoint['inputs'])
        validate_labels(datapoint['labels'])
        validate_meta_info(datapoint['meta_info'], 0)


def test_threedmatch_dataset_determinism():
    """Test that the dataset is deterministic with the same seed."""
    dataset1 = ThreeDMatchDataset(
        data_root='./data/datasets/soft_links/threedmatch',
        split='train',
        num_points=1000,
        base_seed=42,
    )
    
    dataset2 = ThreeDMatchDataset(
        data_root='./data/datasets/soft_links/threedmatch',
        split='train',
        num_points=1000,
        base_seed=42,
    )
    
    if len(dataset1) > 0:
        # Check first datapoint
        data1 = dataset1[0]
        data2 = dataset2[0]
        
        # Check that sampled points are identical
        assert torch.allclose(data1['inputs']['src_pc']['pos'], data2['inputs']['src_pc']['pos'])
        assert torch.allclose(data1['inputs']['tgt_pc']['pos'], data2['inputs']['tgt_pc']['pos'])
        
        # Check that correspondences are identical
        assert torch.equal(data1['inputs']['correspondences'], data2['inputs']['correspondences'])