from typing import Dict, Any
import pytest
import torch
from concurrent.futures import ThreadPoolExecutor
from data.datasets.change_detection_datasets.bi_temporal.urb3dcd_dataset import Urb3DCDDataset


def validate_point_cloud(pc: Dict[str, torch.Tensor], name: str) -> None:
    """Validate a point cloud dictionary."""
    assert isinstance(pc, dict), f"{name} should be a dictionary"
    assert 'pos' in pc, f"{name} should have 'pos' key"
    assert 'feat' in pc, f"{name} should have 'feat' key"

    # Validate pos tensor
    assert isinstance(pc['pos'], torch.Tensor), f"{name}['pos'] should be a torch.Tensor"
    assert pc['pos'].ndim == 2, f"{name}['pos'] should have 2 dimensions (N x 3), got {pc['pos'].ndim}"
    assert pc['pos'].size(1) == 3, f"{name}['pos'] should have 3 features (x,y,z), got {pc['pos'].size(1)}"
    assert torch.is_floating_point(pc['pos']), f"{name}['pos'] should be of dtype torch.float"

    # Validate feat tensor
    assert isinstance(pc['feat'], torch.Tensor), f"{name}['feat'] should be a torch.Tensor"
    assert pc['feat'].ndim == 2, f"{name}['feat'] should have 2 dimensions (N x F), got {pc['feat'].ndim}"
    assert pc['feat'].size(1) == 1, f"{name}['feat'] should have 1 feature (F), got {pc['feat'].size(1)}"
    assert torch.is_floating_point(pc['feat']), f"{name}['feat'] should be of dtype torch.float"


def validate_change_map(change_map: torch.Tensor) -> None:
    """Validate the change map tensor."""
    assert isinstance(change_map, torch.Tensor), "change_map should be a torch.Tensor"
    assert change_map.ndim == 1, "change_map should have 1 dimension (N,)"
    assert change_map.dtype == torch.long, "change_map should be of dtype torch.long"
    unique_values = torch.unique(change_map)
    assert all(val in range(Urb3DCDDataset.NUM_CLASSES) for val in unique_values), \
        f"Unexpected values in change_map: {unique_values}"


def validate_point_count_consistency(pc1: Dict[str, torch.Tensor], change_map: torch.Tensor) -> None:
    """Validate that pc_1 and change_map have the same number of points."""
    assert pc1['pos'].size(0) == change_map.size(0), \
        f"Number of points in pc_1 ({pc1['pos'].size(0)}) does not match " \
        f"number of points in change_map ({change_map.size(0)})"


def validate_inputs(inputs: Dict[str, Any]) -> None:
    """Validate the inputs of a datapoint."""
    assert isinstance(inputs, dict)
    assert set(inputs.keys()) == {'pc_1', 'pc_2'}
    assert isinstance(inputs['pc_1'], dict)
    validate_point_cloud(inputs['pc_1'], 'pc_1')
    assert isinstance(inputs['pc_2'], dict)
    validate_point_cloud(inputs['pc_2'], 'pc_2')


def validate_labels(labels: Dict[str, Any]) -> None:
    """Validate the labels of a datapoint."""
    assert isinstance(labels, dict)
    assert 'change_map' in labels
    assert isinstance(labels['change_map'], torch.Tensor)
    validate_change_map(labels['change_map'])


def validate_meta_info(meta_info: Dict[str, Any], datapoint_idx: int) -> None:
    """Validate the meta_info of a datapoint."""
    assert isinstance(meta_info, dict)
    assert 'point_idx_pc1' in meta_info
    assert 'point_idx_pc2' in meta_info
    assert isinstance(meta_info['point_idx_pc1'], torch.Tensor)
    assert isinstance(meta_info['point_idx_pc2'], torch.Tensor)
    assert meta_info['point_idx_pc1'].dtype == torch.long
    assert meta_info['point_idx_pc2'].dtype == torch.long
    assert 'pc_1_filepath' in meta_info
    assert 'pc_2_filepath' in meta_info
    assert isinstance(meta_info['pc_1_filepath'], str)
    assert isinstance(meta_info['pc_2_filepath'], str)
    assert 'idx' in meta_info, f"meta_info should contain 'idx' key: {meta_info.keys()=}"
    assert meta_info['idx'] == datapoint_idx, f"meta_info['idx'] should match datapoint index: {meta_info['idx']=}, {datapoint_idx=}"


@pytest.fixture
def dataset(request):
    """Fixture for creating a Urb3DCDDataset instance."""
    sample_per_epoch, radius, fix_samples = request.param
    return Urb3DCDDataset(
        data_root="./data/datasets/soft_links/Urb3DCD",
        sample_per_epoch=sample_per_epoch,
        radius=radius,
        fix_samples=fix_samples
    )


@pytest.mark.parametrize('dataset', [
    (100, 100, False),
    (0, 100, False),  # Grid sampling mode
    (100, 100, True),  # Fixed sampling mode
], indirect=True)
def test_urb3dcd_dataset(dataset, max_samples, get_samples_to_test) -> None:
    """Test the Urb3DCDDataset class."""
    print("Dataset initialized.")

    # Verify class labels mapping
    assert len(dataset.INV_OBJECT_LABEL) == dataset.NUM_CLASSES
    assert len(dataset.CLASS_LABELS) == dataset.NUM_CLASSES
    assert all(dataset.CLASS_LABELS[name] == idx for idx, name in dataset.INV_OBJECT_LABEL.items())

    assert len(dataset) > 0

    def validate_datapoint(idx: int) -> None:
        datapoint = dataset[idx]
        inputs = datapoint['inputs']
        labels = datapoint['labels']
        meta_info = datapoint['meta_info']
        validate_inputs(inputs)
        validate_labels(labels)
        validate_point_count_consistency(inputs['pc_2'], labels['change_map'])
        validate_meta_info(meta_info, idx)

    num_samples = get_samples_to_test(len(dataset), max_samples)
    indices = list(range(num_samples))
    with ThreadPoolExecutor() as executor:
        executor.map(validate_datapoint, indices)

    print(f"Testing samples completed.")


def test_fixed_samples_consistency() -> None:
    """Test that fixed sampling mode produces consistent results."""
    dataset = Urb3DCDDataset(
        data_root="./data/datasets/soft_links/Urb3DCD",
        sample_per_epoch=100,
        fix_samples=True
    )

    # Sample twice and verify results are the same
    if len(dataset) > 0:  # Only test if dataset is not empty
        datapoint1 = dataset[0]
        inputs1 = datapoint1['inputs']
        labels1 = datapoint1['labels']
        meta_info1 = datapoint1['meta_info']
        datapoint2 = dataset[0]
        inputs2 = datapoint2['inputs']
        labels2 = datapoint2['labels']
        meta_info2 = datapoint2['meta_info']
        # Test inputs consistency
        assert torch.allclose(inputs1['pc_1']['pos'], inputs2['pc_1']['pos'])
        assert torch.allclose(inputs1['pc_1']['feat'], inputs2['pc_1']['feat'])
        assert torch.allclose(inputs1['pc_2']['pos'], inputs2['pc_2']['pos'])
        assert torch.allclose(inputs1['pc_2']['feat'], inputs2['pc_2']['feat'])

        # Test labels consistency
        assert torch.equal(labels1['change_map'], labels2['change_map'])

        # Test metadata consistency
        assert meta_info1['pc_1_filepath'] == meta_info2['pc_1_filepath']
        assert meta_info1['pc_2_filepath'] == meta_info2['pc_2_filepath']
        assert torch.equal(meta_info1['point_idx_pc1'], meta_info2['point_idx_pc1'])
        assert torch.equal(meta_info1['point_idx_pc2'], meta_info2['point_idx_pc2'])
