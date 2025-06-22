"""Test TEASER++ point cloud registration method."""
import pytest
import torch
from tests.models.utils.point_cloud_data import generate_point_cloud_data
from .common import get_dummy_input, validate_transformation_output


def get_teaser_plusplus_model(voxel_size: float = 0.05, estimate_rotation: bool = True, estimate_scaling: bool = True):
    """Get TEASER++ model instance."""
    from models.point_cloud_registration.classic.teaserplusplus import TeaserPlusPlus
    return TeaserPlusPlus(
        estimate_rotation=estimate_rotation,
        estimate_scaling=estimate_scaling,
        voxel_size=voxel_size
    )


def test_teaser_plusplus_initialization():
    """Test TEASER++ initialization."""
    model = get_teaser_plusplus_model(voxel_size=0.1, estimate_rotation=True, estimate_scaling=False)
    assert model.voxel_size == 0.1
    assert model.estimate_rotation == True
    assert model.estimate_scaling == False


def test_teaser_plusplus_forward_pass_basic():
    """Test basic forward pass."""
    model = get_teaser_plusplus_model()
    model.eval()
    
    dummy_input = get_dummy_input()
    
    with torch.no_grad():
        output = model(dummy_input)
    
    validate_transformation_output(output, batch_size=2)


def test_teaser_plusplus_different_voxel_sizes():
    """Test TEASER++ with different voxel sizes."""
    voxel_sizes = [0.02, 0.05, 0.1, 0.2]
    
    for voxel_size in voxel_sizes:
        model = get_teaser_plusplus_model(voxel_size=voxel_size)
        dummy_input = get_dummy_input()
        
        with torch.no_grad():
            output = model(dummy_input)
        
        validate_transformation_output(output, batch_size=2)


def test_teaser_plusplus_different_estimation_options():
    """Test TEASER++ with different estimation options."""
    options = [
        (True, True),   # Both rotation and scaling
        (True, False),  # Only rotation
        (False, True),  # Only scaling
        (False, False)  # Neither (minimal case)
    ]
    
    for estimate_rotation, estimate_scaling in options:
        model = get_teaser_plusplus_model(
            estimate_rotation=estimate_rotation,
            estimate_scaling=estimate_scaling
        )
        dummy_input = get_dummy_input()
        
        with torch.no_grad():
            output = model(dummy_input)
        
        validate_transformation_output(output, batch_size=2)


def test_teaser_plusplus_different_point_cloud_sizes():
    """Test TEASER++ with different point cloud sizes."""
    model = get_teaser_plusplus_model()
    model.eval()
    
    sizes = [50, 100, 500, 1000]
    
    for size in sizes:
        dummy_input = generate_point_cloud_data(
            batch_size=1, num_points=size, feature_dim=32
        )
        
        with torch.no_grad():
            output = model(dummy_input)
        
        validate_transformation_output(output, batch_size=1)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_teaser_plusplus_different_batch_sizes(batch_size):
    """Test TEASER++ with different batch sizes."""
    model = get_teaser_plusplus_model()
    model.eval()
    
    dummy_input = generate_point_cloud_data(batch_size=batch_size)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    validate_transformation_output(output, batch_size=batch_size)


@pytest.mark.parametrize("voxel_size,estimate_rotation,estimate_scaling", [
    (0.02, True, True),
    (0.05, True, False),
    (0.1, False, True),
    (0.15, False, False)
])
def test_teaser_plusplus_parameter_combinations(voxel_size, estimate_rotation, estimate_scaling):
    """Test TEASER++ with different parameter combinations."""
    model = get_teaser_plusplus_model(
        voxel_size=voxel_size,
        estimate_rotation=estimate_rotation,
        estimate_scaling=estimate_scaling
    )
    dummy_input = get_dummy_input()
    
    with torch.no_grad():
        output = model(dummy_input)
    
    validate_transformation_output(output, batch_size=2)


def test_teaser_plusplus_computational_efficiency():
    """Test that TEASER++ completes in reasonable time."""
    import time
    
    model = get_teaser_plusplus_model(voxel_size=0.1)
    
    # Test with small point clouds for speed
    dummy_input = generate_point_cloud_data(
        batch_size=1, num_points=50, feature_dim=32
    )
    
    model.eval()
    start_time = time.time()
    
    with torch.no_grad():
        output = model(dummy_input)
    
    end_time = time.time()
    computation_time = end_time - start_time
    
    # Should complete within reasonable time (5 seconds)
    assert computation_time < 5.0, f"TEASER++ took too long: {computation_time:.2f}s"
    assert output is not None