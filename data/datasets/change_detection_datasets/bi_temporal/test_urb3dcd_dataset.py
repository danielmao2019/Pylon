import pytest
import torch
from .urb3dcd_dataset import Urb3DCDDataset


@pytest.mark.parametrize("dataset_params", [
    {"sample_per_epoch": 100, "radius": 2, "fix_samples": False},
    {"sample_per_epoch": 0, "radius": 2, "fix_samples": False},  # Grid sampling mode
    {"sample_per_epoch": 100, "radius": 2, "fix_samples": True},  # Fixed sampling mode
])
def test_urb3dcd_dataset(dataset_params, tmp_path):
    # Create a dataset instance
    dataset = Urb3DCDDataset(
        data_root=str(tmp_path),  # Use temporary directory for testing
        **dataset_params
    )
    
    assert isinstance(dataset, torch.utils.data.Dataset)
    assert dataset.NUM_CLASSES == 7
    assert len(dataset.INPUT_NAMES) == 4
    assert len(dataset.LABEL_NAMES) == 1
    assert dataset.IGNORE_LABEL == -1

    # Verify class labels mapping
    assert len(dataset.INV_OBJECT_LABEL) == dataset.NUM_CLASSES
    assert len(dataset.CLASS_LABELS) == dataset.NUM_CLASSES
    assert all(dataset.CLASS_LABELS[name] == idx for idx, name in dataset.INV_OBJECT_LABEL.items())


def _validate_point_cloud(pc: torch.Tensor, name: str):
    """Validate a point cloud tensor."""
    assert isinstance(pc, torch.Tensor), f"{name} should be a torch.Tensor"
    assert pc.ndim == 2, f"{name} should have 2 dimensions (N x 3), got {pc.ndim}"
    assert pc.size(1) == 3, f"{name} should have 3 features (x,y,z), got {pc.size(1)}"
    assert pc.dtype == torch.float, f"{name} should be of dtype torch.float"


def _validate_kdtree(kdtree, name: str):
    """Validate a KDTree object."""
    from sklearn.neighbors import KDTree
    assert isinstance(kdtree, KDTree), f"{name} should be a sklearn.neighbors.KDTree"


def _validate_change_map(change_map: torch.Tensor):
    """Validate the change map tensor."""
    assert isinstance(change_map, torch.Tensor), "change_map should be a torch.Tensor"
    assert change_map.dtype == torch.long, "change_map should be of dtype torch.long"
    unique_values = torch.unique(change_map)
    assert all(val in range(7) for val in unique_values), f"Unexpected values in change_map: {unique_values}"


@pytest.mark.parametrize("radius", [1.0, 2.0, 3.0])
def test_sampling_radius(radius, tmp_path):
    """Test that sampling radius parameter is respected."""
    dataset = Urb3DCDDataset(
        data_root=str(tmp_path),
        sample_per_epoch=100,
        radius=radius
    )
    assert dataset._radius == radius


def test_fixed_samples_consistency(tmp_path):
    """Test that fixed sampling mode produces consistent results."""
    dataset = Urb3DCDDataset(
        data_root=str(tmp_path),
        sample_per_epoch=100,
        fix_samples=True
    )
    
    # Sample twice and verify results are the same
    if len(dataset) > 0:  # Only test if dataset is not empty
        first_sample = dataset.get(0)
        second_sample = dataset.get(0)
        
        if first_sample is not None and second_sample is not None:
            assert torch.allclose(first_sample['pc_0'], second_sample['pc_0'])
            assert torch.allclose(first_sample['pc_1'], second_sample['pc_1'])
            assert torch.equal(first_sample['change_map'], second_sample['change_map'])


def test_grid_sampling_mode(tmp_path):
    """Test grid sampling mode (sample_per_epoch=0)."""
    dataset = Urb3DCDDataset(
        data_root=str(tmp_path),
        sample_per_epoch=0,
        radius=2
    )
    
    # Verify that grid sampling centers are created
    assert hasattr(dataset, 'grid_regular_centers')
    if len(dataset.grid_regular_centers) > 0:
        assert isinstance(dataset.grid_regular_centers, torch.Tensor)
        assert dataset.grid_regular_centers.size(1) == 4  # x, y, z, area_index
