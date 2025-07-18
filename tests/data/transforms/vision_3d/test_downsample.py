import torch
import pytest
from data.transforms.vision_3d.downsample import DownSample


@pytest.fixture
def sample_point_cloud():
    # Create a sample point cloud with multiple fields
    num_points = 1000
    pc = {
        'pos': torch.randn(num_points, 3),  # 3D positions
        'rgb': torch.rand(num_points, 3),   # RGB colors
        'intensity': torch.rand(num_points), # Intensity values
        'feat': torch.randn(num_points, 4)  # Some features
    }
    return pc


@pytest.fixture
def multiple_point_clouds():
    # Create multiple point clouds with different sizes and fields
    pcs = []

    # First PC: standard fields
    pc1 = {
        'pos': torch.randn(1000, 3),
        'rgb': torch.rand(1000, 3),
        'intensity': torch.rand(1000)
    }
    pcs.append(pc1)

    # Second PC: different size, different fields
    pc2 = {
        'pos': torch.randn(500, 3),
        'feat': torch.randn(500, 4),
        'label': torch.randint(0, 10, (500,))
    }
    pcs.append(pc2)

    # Third PC: empty point cloud
    pc3 = {
        'pos': torch.empty(0, 3),
        'rgb': torch.empty(0, 3)
    }
    pcs.append(pc3)

    return pcs


def test_downsample_basic(sample_point_cloud):
    # Test basic downsampling with a reasonable voxel size
    voxel_size = 0.1
    downsample = DownSample(voxel_size=voxel_size)

    # Apply downsampling
    result = downsample(sample_point_cloud)

    # Check that all fields are present
    assert set(result.keys()) == set(sample_point_cloud.keys()) | {'indices'}

    # Check that all tensors have the same number of points
    num_points = result['pos'].shape[0]
    for key, value in result.items():
        assert isinstance(value, torch.Tensor)
        assert value.shape[0] == num_points

    # Check that the number of points decreased
    assert num_points <= sample_point_cloud['pos'].shape[0]

    # Check that the data types are preserved
    for key in result.keys():
        if key == 'indices':
            continue
        assert result[key].dtype == sample_point_cloud[key].dtype


def test_downsample_device_consistency(sample_point_cloud):
    # Test that downsampling works on GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # Move point cloud to GPU
        gpu_pc = {k: v.to(device) for k, v in sample_point_cloud.items()}

        downsample = DownSample(voxel_size=0.1)
        result = downsample(gpu_pc)

        # Check that all tensors are on the same device
        for key, value in result.items():
            assert value.device.type == device.type


def test_downsample_voxel_size_effect(sample_point_cloud):
    # Test that larger voxel size results in fewer points
    downsample_small = DownSample(voxel_size=0.1)
    downsample_large = DownSample(voxel_size=0.5)

    result_small = downsample_small(sample_point_cloud)
    result_large = downsample_large(sample_point_cloud)

    # Larger voxel size should result in fewer points
    assert result_large['pos'].shape[0] <= result_small['pos'].shape[0]


def test_downsample_empty_point_cloud():
    # Test with an empty point cloud
    empty_pc = {
        'pos': torch.empty(0, 3),
        'rgb': torch.empty(0, 3),
        'intensity': torch.empty(0)
    }

    downsample = DownSample(voxel_size=0.1)
    result = downsample(empty_pc)

    # Should return empty tensors
    for key, value in result.items():
        assert value.shape[0] == 0


def test_downsample_single_point():
    # Test with a single point
    single_point_pc = {
        'pos': torch.randn(1, 3),
        'rgb': torch.rand(1, 3),
        'intensity': torch.rand(1)
    }

    downsample = DownSample(voxel_size=0.1)
    result = downsample(single_point_pc)

    # Should keep the single point
    for key, value in result.items():
        assert value.shape[0] == 1


def test_downsample_multiple_point_clouds(multiple_point_clouds):
    # Test downsampling multiple point clouds
    downsample = DownSample(voxel_size=0.1)
    results = downsample(*multiple_point_clouds)

    # Check that we got the same number of point clouds back
    assert len(results) == len(multiple_point_clouds)

    # Check each point cloud
    for original_pc, result_pc in zip(multiple_point_clouds, results):
        # Check that all fields are present
        assert set(result_pc.keys()) == set(original_pc.keys()) | {'indices'}

        # Check that all tensors have the same number of points
        num_points = result_pc['pos'].shape[0]
        for key, value in result_pc.items():
            assert isinstance(value, torch.Tensor)
            assert value.shape[0] == num_points

        # Check that the number of points decreased (unless it was empty)
        if original_pc['pos'].shape[0] > 0:
            assert num_points <= original_pc['pos'].shape[0]
        else:
            assert num_points == 0

        # Check that the data types are preserved
        for key in result_pc.keys():
            if key == 'indices':
                continue
            assert result_pc[key].dtype == original_pc[key].dtype


@pytest.mark.parametrize("invalid_voxel_size", [
    0,      # Zero voxel size
    -1,     # Negative voxel size
    -0.1,   # Negative float voxel size
    "0.1",  # String instead of number
    None,   # None value
    [],     # Empty list
    {},     # Empty dict
])
def test_downsample_invalid_voxel_size(invalid_voxel_size):
    # Test with invalid voxel size
    with pytest.raises(AssertionError):
        DownSample(voxel_size=invalid_voxel_size)
