import pytest
import torch
from .urb3dcd_dataset import Urb3DCDDataset


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
    assert all(val in range(urb3dcd_dataset.NUM_CLASSES) for val in unique_values), \
        f"Unexpected values in change_map: {unique_values}"


def _validate_point_count_consistency(pc1: torch.Tensor, change_map: torch.Tensor):
    """Validate that pc_1 and change_map have the same number of points."""
    assert pc1.size(0) == change_map.size(0), (
        f"Number of points in pc_1 ({pc1.size(0)}) does not match "
        f"number of points in change_map ({change_map.size(0)})"
    )


@pytest.mark.parametrize("dataset_params", [
    {"sample_per_epoch": 100, "radius": 2, "fix_samples": False},
    {"sample_per_epoch": 0, "radius": 2, "fix_samples": False},  # Grid sampling mode
    {"sample_per_epoch": 100, "radius": 2, "fix_samples": True},  # Fixed sampling mode
])
def test_urb3dcd_dataset(dataset_params):
    """Test the Urb3DCDDataset class."""
    # Create a dataset instance
    print("Initializing dataset...")
    dataset = Urb3DCDDataset(
        data_root="./data/datasets/soft_links/Urb3DCD",
        **dataset_params
    )
    print("Dataset initialized.")

    # Verify class labels mapping
    assert len(dataset.INV_OBJECT_LABEL) == dataset.NUM_CLASSES
    assert len(dataset.CLASS_LABELS) == dataset.NUM_CLASSES
    assert all(dataset.CLASS_LABELS[name] == idx for idx, name in dataset.INV_OBJECT_LABEL.items())

    assert len(dataset) > 0
    # Test first few samples
    print(f"Testing samples...")
    for idx in range(min(3, len(dataset))):
        inputs, labels, meta_info = dataset[idx]
        
        # Validate point clouds
        _validate_point_cloud(inputs['pc_0'], 'pc_0')
        _validate_point_cloud(inputs['pc_1'], 'pc_1')
        
        # Validate KDTrees
        _validate_kdtree(inputs['kdtree_0'], 'kdtree_0')
        _validate_kdtree(inputs['kdtree_1'], 'kdtree_1')
        
        # Validate change map
        _validate_change_map(labels['change_map'])
        
        # Validate point count consistency
        _validate_point_count_consistency(inputs['pc_1'], labels['change_map'])
        
        # Test point indices if present
        assert 'point_idx_pc0' in meta_info
        assert 'point_idx_pc1' in meta_info
        assert isinstance(meta_info['point_idx_pc0'], torch.Tensor)
        assert isinstance(meta_info['point_idx_pc1'], torch.Tensor)
        assert meta_info['point_idx_pc0'].dtype == torch.long
        assert meta_info['point_idx_pc1'].dtype == torch.long


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
        inputs1, labels1, _ = dataset[0]
        inputs2, labels2, _ = dataset[0]
        
        assert torch.allclose(inputs1['pc_0'], inputs2['pc_0'])
        assert torch.allclose(inputs1['pc_1'], inputs2['pc_1'])
        assert torch.equal(labels1['change_map'], labels2['change_map'])


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
        assert isinstance(dataset.grid_regular_centers, dict)
        assert 'pos' in dataset.grid_regular_centers
        assert 'idx' in dataset.grid_regular_centers
        assert 'change_map' in dataset.grid_regular_centers
