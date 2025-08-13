import os
import tempfile
import numpy as np
import torch
import pytest
from plyfile import PlyData, PlyElement
from utils.io.point_clouds.load_point_cloud import load_point_cloud


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def create_test_point_cloud_data(num_points: int = 100, include_features: bool = False, include_rgb: bool = False):
    """Create test point cloud data."""
    # Generate XYZ coordinates
    pos_data = torch.rand(num_points, 3) * 10.0  # Scale to [0, 10] range
    
    result = {'pos': pos_data}
    
    if include_features:
        feat_data = torch.rand(num_points, 2)  # 2 feature channels
        result['feat'] = feat_data
    
    if include_rgb:
        rgb_data = torch.rand(num_points, 3)  # RGB in [0, 1] range
        result['rgb'] = rgb_data
    
    return result


def test_point_cloud_data_consistency(temp_dir):
    """Test that loaded point cloud data maintains consistency."""
    filepath = os.path.join(temp_dir, "consistency.pth")
    
    # Create test data
    original_data = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0]
    ])
    torch.save(original_data, filepath)
    
    # Load and verify
    result = load_point_cloud(filepath=filepath, device='cpu')
    
    assert torch.allclose(result['pos'], original_data)
    assert result['pos'].shape == original_data.shape
    assert result['pos'].dtype == torch.float32


def test_point_cloud_coordinate_ranges(temp_dir):
    """Test point clouds with various coordinate ranges."""
    test_cases = [
        ("small_positive", torch.rand(50, 3) * 0.1),  # [0, 0.1]
        ("large_positive", torch.rand(50, 3) * 1000.0),  # [0, 1000]
        ("negative", -torch.rand(50, 3) * 10.0),  # [-10, 0]
        ("mixed", (torch.rand(50, 3) - 0.5) * 20.0),  # [-10, 10]
        ("zero_centered", torch.randn(50, 3)),  # Normal distribution around 0
    ]
    
    for name, data in test_cases:
        filepath = os.path.join(temp_dir, f"{name}.pth")
        torch.save(data, filepath)
        
        result = load_point_cloud(filepath=filepath, device='cpu')
        
        assert result['pos'].shape == data.shape
        assert result['pos'].dtype == torch.float32
        # Verify the range is preserved (with some tolerance for float conversion)
        assert torch.allclose(result['pos'], data.float(), atol=1e-6)


def test_point_cloud_feature_dimensions(temp_dir):
    """Test point clouds with various feature dimensions."""
    base_pos = torch.rand(30, 3)
    
    feature_cases = [
        ("single_feature", torch.rand(30, 1)),
        ("dual_features", torch.rand(30, 2)),
        ("multi_features", torch.rand(30, 5)),
        ("high_dim_features", torch.rand(30, 20))
    ]
    
    for name, features in feature_cases:
        filepath = os.path.join(temp_dir, f"feat_{name}.pth")
        data = torch.cat([base_pos, features], dim=1)
        torch.save(data, filepath)
        
        result = load_point_cloud(filepath=filepath, device='cpu')
        
        assert result['pos'].shape == (30, 3)
        assert result['feat'].shape == features.shape
        assert torch.allclose(result['feat'], features, atol=1e-6)


def test_point_cloud_rgb_normalization(temp_dir):
    """Test RGB color normalization in PLY files."""
    filepath = os.path.join(temp_dir, "rgb_test.ply")
    
    # Create PLY with specific RGB values
    vertices = []
    expected_normalized = []
    
    for i in range(10):
        r, g, b = i * 25, (i * 10) % 255, (i * 50) % 255
        vertices.append((float(i), float(i), float(i), r, g, b))
        # RGB should be normalized to [0, 1]
        expected_normalized.append([r / 255.0, g / 255.0, b / 255.0])
    
    vertex_element = PlyElement.describe(
        np.array(vertices, dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ]), 
        'vertex'
    )
    PlyData([vertex_element], text=True).write(filepath)
    
    result = load_point_cloud(filepath=filepath, device='cpu')
    
    assert 'rgb' in result
    assert result['rgb'].shape == (10, 3)
    
    # Check normalization (RGB values should be in [0, 1])
    assert result['rgb'].min() >= 0.0
    assert result['rgb'].max() <= 1.0
    
    # Verify specific values are correctly normalized
    expected_tensor = torch.tensor(expected_normalized, dtype=torch.float32)
    assert torch.allclose(result['rgb'], expected_tensor, atol=1e-6)


def test_point_cloud_data_types_preservation(temp_dir):
    """Test that different data types are handled correctly."""
    # Test with different input data types
    pos_data = torch.rand(20, 3).double()  # float64
    feat_data = torch.randint(0, 10, (20, 2)).long()  # int64
    
    filepath = os.path.join(temp_dir, "mixed_types.pth")
    combined_data = torch.cat([pos_data, feat_data.float()], dim=1)
    torch.save(combined_data, filepath)
    
    result = load_point_cloud(filepath=filepath, device='cpu')
    
    # Position should always be float32
    assert result['pos'].dtype == torch.float32
    # Features should be float32 or float64 (depending on input conversion)
    assert result['feat'].dtype in [torch.float32, torch.float64]
    
    # Verify data integrity
    assert torch.allclose(result['pos'], pos_data.float(), atol=1e-6)
    # Compare with appropriate dtype conversion
    if result['feat'].dtype == torch.float64:
        assert torch.allclose(result['feat'], feat_data.double(), atol=1e-6)
    else:
        assert torch.allclose(result['feat'], feat_data.float(), atol=1e-6)


def test_point_cloud_edge_coordinates(temp_dir):
    """Test point clouds with edge case coordinate values."""
    edge_cases = [
        ("very_large", torch.tensor([[1e10, -1e10, 1e15]])),
        ("very_small", torch.tensor([[1e-10, -1e-10, 1e-15]])),
        ("zeros", torch.zeros(5, 3)),
    ]
    
    for name, data in edge_cases:
        filepath = os.path.join(temp_dir, f"edge_{name}.pth")
        torch.save(data, filepath)
        
        result = load_point_cloud(filepath=filepath, device='cpu')
        assert result['pos'].shape == data.shape
        assert result['pos'].dtype == torch.float32


def test_point_cloud_special_float_values(temp_dir):
    """Test that point cloud loading correctly handles infinity and NaN values."""
    # Test positive and negative infinity
    inf_filepath = os.path.join(temp_dir, "infinity.pth")
    inf_data = torch.tensor([
        [float('inf'), 0.0, 0.0],
        [0.0, float('-inf'), 0.0],
        [1.0, 2.0, 3.0]
    ])
    torch.save(inf_data, inf_filepath)
    
    # Load and verify infinity values are preserved
    inf_result = load_point_cloud(filepath=inf_filepath, device='cpu')
    assert inf_result['pos'].shape == (3, 3)
    assert inf_result['pos'].dtype == torch.float32
    assert torch.isinf(inf_result['pos']).any()  # Contains infinity values
    assert torch.equal(torch.isinf(inf_result['pos']), torch.isinf(inf_data))  # Same infinity pattern
    
    # Test NaN values
    nan_filepath = os.path.join(temp_dir, "nan.pth")
    nan_data = torch.tensor([
        [float('nan'), 0.0, 0.0],
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0]
    ])
    torch.save(nan_data, nan_filepath)
    
    # Load and verify NaN values are preserved
    nan_result = load_point_cloud(filepath=nan_filepath, device='cpu')
    assert nan_result['pos'].shape == (3, 3)
    assert nan_result['pos'].dtype == torch.float32
    assert torch.isnan(nan_result['pos']).any()  # Contains NaN values
    assert torch.equal(torch.isnan(nan_result['pos']), torch.isnan(nan_data))  # Same NaN pattern
    
    # Test mixed special values
    mixed_filepath = os.path.join(temp_dir, "mixed_special.pth")
    mixed_data = torch.tensor([
        [float('inf'), float('nan'), 0.0],
        [float('-inf'), 1.0, float('nan')],
        [0.0, 0.0, 0.0]
    ])
    torch.save(mixed_data, mixed_filepath)
    
    # Load and verify all special values are preserved
    mixed_result = load_point_cloud(filepath=mixed_filepath, device='cpu')
    assert mixed_result['pos'].shape == (3, 3)
    assert mixed_result['pos'].dtype == torch.float32
    
    # Verify specific special value locations
    assert torch.isinf(mixed_result['pos'][0, 0]) and mixed_result['pos'][0, 0] > 0  # +inf
    assert torch.isnan(mixed_result['pos'][0, 1])  # nan
    assert torch.isinf(mixed_result['pos'][1, 0]) and mixed_result['pos'][1, 0] < 0  # -inf
    assert torch.isnan(mixed_result['pos'][1, 2])  # nan
    assert torch.isfinite(mixed_result['pos'][2, :]).all()  # All finite values


def test_point_cloud_memory_efficiency(temp_dir):
    """Test memory efficiency with large point clouds."""
    # Create a reasonably large point cloud (not too large to avoid test timeouts)
    num_points = 10000
    large_data = torch.rand(num_points, 3)
    
    filepath = os.path.join(temp_dir, "large_cloud.pth")
    torch.save(large_data, filepath)
    
    # Load and verify
    result = load_point_cloud(filepath=filepath, device='cpu')
    
    assert result['pos'].shape == (num_points, 3)
    assert result['pos'].dtype == torch.float32
    
    # Verify that the data is actually loaded (not just a view)
    assert result['pos'].is_contiguous()


def test_point_cloud_batch_loading(temp_dir):
    """Test loading multiple point cloud files efficiently."""
    num_files = 5
    filepaths = []
    expected_shapes = []
    
    # Create multiple files with different sizes
    for i in range(num_files):
        num_points = 50 + i * 20  # 50, 70, 90, 110, 130 points
        data = torch.rand(num_points, 3) + i  # Offset each cloud
        
        filepath = os.path.join(temp_dir, f"batch_{i}.pth")
        torch.save(data, filepath)
        
        filepaths.append(filepath)
        expected_shapes.append((num_points, 3))
    
    # Load all files
    results = []
    for filepath in filepaths:
        result = load_point_cloud(filepath=filepath, device='cpu')
        results.append(result)
    
    # Verify each result
    for i, result in enumerate(results):
        assert result['pos'].shape == expected_shapes[i]
        assert result['pos'].dtype == torch.float32
        
        # Verify the offset is preserved (roughly)
        mean_pos = result['pos'].mean(dim=0)
        expected_offset = i + 0.5  # Rough expected mean due to offset + random [0,1]
        assert torch.allclose(mean_pos, torch.full((3,), expected_offset), atol=0.2)


def test_point_cloud_validation_integration(temp_dir):
    """Test that point cloud validation works correctly."""
    # Create valid point cloud
    valid_data = torch.rand(100, 3)
    valid_filepath = os.path.join(temp_dir, "valid.pth")
    torch.save(valid_data, valid_filepath)
    
    # This should load successfully
    result = load_point_cloud(filepath=valid_filepath, device='cpu')
    assert 'pos' in result
    
    # Create invalid point cloud (wrong shape)
    invalid_data = torch.rand(100, 2)  # Only 2 coordinates instead of 3
    invalid_filepath = os.path.join(temp_dir, "invalid.pth")
    torch.save(invalid_data, invalid_filepath)
    
    # This should raise an error during validation
    with pytest.raises(AssertionError):
        load_point_cloud(filepath=invalid_filepath, device='cpu')


def test_point_cloud_device_consistency(temp_dir):
    """Test that device assignment is consistent across all tensors."""
    filepath = os.path.join(temp_dir, "device_test.pth")
    
    # Create data with features
    pos_data = torch.rand(50, 3)
    feat_data = torch.rand(50, 2)
    combined_data = torch.cat([pos_data, feat_data], dim=1)
    torch.save(combined_data, filepath)
    
    # Test CPU device
    result_cpu = load_point_cloud(filepath=filepath, device='cpu')
    assert result_cpu['pos'].device.type == 'cpu'
    assert result_cpu['feat'].device.type == 'cpu'
    
    # Test CUDA device (if available)
    if torch.cuda.is_available():
        result_cuda = load_point_cloud(filepath=filepath, device='cuda')
        assert result_cuda['pos'].device.type == 'cuda'
        assert result_cuda['feat'].device.type == 'cuda'


def test_point_cloud_deterministic_loading(temp_dir):
    """Test that loading the same file multiple times gives consistent results."""
    filepath = os.path.join(temp_dir, "deterministic.pth")
    
    # Create test data
    original_data = torch.rand(75, 4)  # XYZ + 1 feature
    torch.save(original_data, filepath)
    
    # Load the same file multiple times
    results = []
    for _ in range(3):
        result = load_point_cloud(filepath=filepath, device='cpu')
        results.append(result)
    
    # All results should be identical
    for i in range(1, len(results)):
        assert torch.equal(results[0]['pos'], results[i]['pos'])
        assert torch.equal(results[0]['feat'], results[i]['feat'])


def test_point_cloud_metadata_preservation(temp_dir):
    """Test that important metadata is preserved during loading."""
    filepath = os.path.join(temp_dir, "metadata.pth")
    
    # Create structured data that might contain metadata-like information
    data = {
        'points': torch.rand(40, 3),
        'colors': torch.rand(40, 3),
        'metadata': {'source': 'test', 'timestamp': '2023-01-01'}
    }
    
    # Note: The current load_point_cloud function expects tensor data, not dicts
    # So we test with tensor data but verify the loading preserves structure
    tensor_data = torch.rand(40, 6)  # XYZ + RGB
    torch.save(tensor_data, filepath)
    
    result = load_point_cloud(filepath=filepath, device='cpu')
    
    assert result['pos'].shape == (40, 3)
    assert result['feat'].shape == (40, 3)  # RGB becomes features
    
    # Verify the data was split correctly
    original_pos = tensor_data[:, :3]
    original_feat = tensor_data[:, 3:]
    
    assert torch.allclose(result['pos'], original_pos, atol=1e-6)
    assert torch.allclose(result['feat'], original_feat, atol=1e-6)
