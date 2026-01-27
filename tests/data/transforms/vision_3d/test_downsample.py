import pytest
import torch

from data.structures.three_d.point_cloud.point_cloud import PointCloud
from data.transforms.vision_3d.downsample import DownSample


@pytest.fixture
def sample_point_cloud():
    # Create a sample point cloud with multiple fields
    num_points = 1000
    data = {
        'xyz': torch.randn(num_points, 3),
        'rgb': torch.rand(num_points, 3),
        'intensity': torch.rand(num_points),
        'feat': torch.randn(num_points, 4),
    }
    return PointCloud(data=data)


@pytest.fixture
def multiple_point_clouds():
    # Create multiple point clouds with different sizes and fields
    pcs = []

    # First PC: standard fields
    data1 = {
        'xyz': torch.randn(1000, 3),
        'rgb': torch.rand(1000, 3),
        'intensity': torch.rand(1000),
    }
    pcs.append(PointCloud(data=data1))

    # Second PC: different size, different fields
    data2 = {
        'xyz': torch.randn(500, 3),
        'feat': torch.randn(500, 4),
        'label': torch.randint(0, 10, (500,)),
    }
    pcs.append(PointCloud(data=data2))

    # Third PC: small point cloud (not empty, since empty is invalid)
    data3 = {
        'xyz': torch.randn(10, 3),
        'rgb': torch.rand(10, 3),
    }
    pcs.append(PointCloud(data=data3))

    return pcs


def test_downsample_basic(sample_point_cloud):
    # Test basic downsampling with a reasonable voxel size
    voxel_size = 0.1
    downsample = DownSample(voxel_size=voxel_size)

    # Apply downsampling
    result = downsample(sample_point_cloud)

    # Check that all fields are present
    assert result.field_names() == ('xyz', 'rgb', 'intensity', 'feat', 'indices')

    # Check that all tensors have the same number of points
    num_points = result.num_points
    assert result.rgb.shape[0] == num_points
    assert result.intensity.shape[0] == num_points
    assert result.feat.shape[0] == num_points

    # Check that the number of points decreased
    assert num_points <= sample_point_cloud.num_points

    # Check that the data types are preserved
    assert result.xyz.dtype == sample_point_cloud.xyz.dtype
    assert result.rgb.dtype == sample_point_cloud.rgb.dtype
    assert result.intensity.dtype == sample_point_cloud.intensity.dtype


def test_downsample_device_consistency(sample_point_cloud):
    # Test that downsampling works on GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_pc = PointCloud(
            xyz=sample_point_cloud.xyz.to(device),
            data={
                'rgb': sample_point_cloud.rgb.to(device),
                'intensity': sample_point_cloud.intensity.to(device),
                'feat': sample_point_cloud.feat.to(device),
            },
        )

        downsample = DownSample(voxel_size=0.1)
        result = downsample(gpu_pc)

        # Check that all tensors are on the same device
        assert result.xyz.device.type == device.type
        assert result.rgb.device.type == device.type
        assert result.intensity.device.type == device.type


def test_downsample_voxel_size_effect(sample_point_cloud):
    # Test that larger voxel size results in fewer points
    downsample_small = DownSample(voxel_size=0.1)
    downsample_large = DownSample(voxel_size=0.5)

    result_small = downsample_small(sample_point_cloud)
    result_large = downsample_large(sample_point_cloud)

    # Larger voxel size should result in fewer points
    assert result_large.num_points <= result_small.num_points


def test_downsample_empty_point_cloud():
    # Test with an empty point cloud (should raise assertion error)
    with pytest.raises(AssertionError):
        PointCloud(
            xyz=torch.empty(0, 3),
            data={
                'rgb': torch.empty(0, 3),
                'intensity': torch.empty(0),
            },
        )


def test_downsample_single_point():
    # Test with a single point
    single_point_pc = PointCloud(data={'xyz': torch.randn(1, 3), 'rgb': torch.rand(1, 3), 'intensity': torch.rand(1)})

    downsample = DownSample(voxel_size=0.1)
    result = downsample(single_point_pc)

    # Should keep the single point
    assert result.num_points == 1
    assert result.rgb.shape[0] == 1
    assert result.intensity.shape[0] == 1


def test_downsample_multiple_point_clouds(multiple_point_clouds):
    # Test downsampling multiple point clouds
    downsample = DownSample(voxel_size=0.1)
    results = downsample(*multiple_point_clouds)

    # Check that we got the same number of point clouds back
    assert len(results) == len(multiple_point_clouds)

    # Check each point cloud
    for original_pc, result_pc in zip(multiple_point_clouds, results, strict=True):
        # Check that all fields are present
        assert 'indices' in result_pc.field_names()

        # Check that all tensors have the same number of points
        num_points = result_pc.num_points

        # Check that the number of points decreased or stayed the same
        assert num_points <= original_pc.num_points

        # Check that the data types are preserved
        assert result_pc.xyz.dtype == original_pc.xyz.dtype


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
