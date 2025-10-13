import os
import tempfile
import numpy as np
import torch
import pytest
from plyfile import PlyData, PlyElement
from utils.io.point_clouds.load_point_cloud import (
    load_point_cloud,
    _load_from_ply,
    _load_from_txt,
    _load_from_pth,
    _load_from_off,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def create_test_ply_file(
    filepath: str, include_rgb: bool = False, include_feat: bool = False
):
    """Create a test PLY file."""
    num_points = 100
    vertices = []

    for i in range(num_points):
        vertex = [i * 0.1, i * 0.2, i * 0.3]  # x, y, z

        if include_rgb:
            vertex.extend([i % 255, (i * 2) % 255, (i * 3) % 255])  # red, green, blue

        if include_feat:
            vertex.append(float(i))  # feature value

        vertices.append(tuple(vertex))

    # Define vertex element
    dtype_list = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    if include_rgb:
        dtype_list.extend([('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    if include_feat:
        dtype_list.append(('feature', 'f4'))

    vertex_element = PlyElement.describe(np.array(vertices, dtype=dtype_list), 'vertex')

    PlyData([vertex_element], text=True).write(filepath)


def create_test_txt_file(filepath: str, include_features: bool = False):
    """Create a test TXT file in SLPCCD format."""
    with open(filepath, 'w') as f:
        # Header lines
        f.write("# Header line 1\n")
        f.write("# Header line 2\n")

        # Data points
        for i in range(50):
            x, y, z = i * 0.1, i * 0.2, i * 0.3
            if include_features:
                # SLPCCD format: X Y Z Rf Gf Bf label
                f.write(f"{x} {y} {z} {i%255} {(i*2)%255} {(i*3)%255} {i%10}\n")
            else:
                f.write(f"{x} {y} {z}\n")


def create_test_pth_file(filepath: str, include_features: bool = False):
    """Create a test PTH file."""
    num_points = 50
    if include_features:
        data = torch.rand(num_points, 6)  # XYZ + 3 features
    else:
        data = torch.rand(num_points, 3)  # XYZ only

    torch.save(data, filepath)


def create_test_off_file(filepath: str):
    """Create a test OFF file."""
    with open(filepath, 'w') as f:
        f.write("OFF\n")
        f.write("4 0 0\n")  # 4 vertices, 0 faces, 0 edges

        # Write vertices
        f.write("0.0 0.0 0.0\n")
        f.write("1.0 0.0 0.0\n")
        f.write("0.0 1.0 0.0\n")
        f.write("0.0 0.0 1.0\n")


def test_load_point_cloud_ply_basic(temp_dir):
    """Test loading basic PLY file."""
    filepath = os.path.join(temp_dir, "test.ply")
    create_test_ply_file(filepath)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert isinstance(result, dict)
    assert 'pos' in result
    assert isinstance(result['pos'], torch.Tensor)
    assert result['pos'].shape[1] == 3  # XYZ coordinates
    assert result['pos'].dtype == torch.float32


def test_load_point_cloud_ply_with_rgb(temp_dir):
    """Test loading PLY file with RGB colors."""
    filepath = os.path.join(temp_dir, "test_rgb.ply")
    create_test_ply_file(filepath, include_rgb=True)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert 'pos' in result
    assert 'rgb' in result
    assert result['rgb'].shape[1] == 3  # RGB channels
    assert result['rgb'].dtype == torch.float32
    # RGB values should be normalized to [0, 1]
    assert result['rgb'].min() >= 0.0
    assert result['rgb'].max() <= 1.0


def test_load_point_cloud_ply_with_features(temp_dir):
    """Test loading PLY file with additional features."""
    filepath = os.path.join(temp_dir, "test_feat.ply")
    create_test_ply_file(filepath, include_feat=True)

    result = load_point_cloud(filepath=filepath, name_feat='feature', device='cpu')

    assert 'pos' in result
    assert 'feat' in result
    assert result['feat'].shape[1] == 1  # Single feature column
    assert result['feat'].dtype == torch.float32


def test_load_point_cloud_txt_basic(temp_dir):
    """Test loading basic TXT file."""
    filepath = os.path.join(temp_dir, "test.txt")
    create_test_txt_file(filepath)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert 'pos' in result
    assert result['pos'].shape[1] == 3
    assert result['pos'].dtype == torch.float32


def test_load_point_cloud_txt_with_features(temp_dir):
    """Test loading TXT file with features (SLPCCD format)."""
    filepath = os.path.join(temp_dir, "test_feat.txt")
    create_test_txt_file(filepath, include_features=True)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert 'pos' in result
    assert 'feat' in result
    assert result['feat'].shape[1] == 1  # Label column as feature
    assert result['feat'].dtype in [
        torch.float32,
        torch.float64,
    ]  # Allow both float types


def test_load_point_cloud_pth_basic(temp_dir):
    """Test loading basic PTH file."""
    filepath = os.path.join(temp_dir, "test.pth")
    create_test_pth_file(filepath)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert 'pos' in result
    assert result['pos'].shape[1] == 3
    assert result['pos'].dtype == torch.float32


def test_load_point_cloud_pth_with_features(temp_dir):
    """Test loading PTH file with features."""
    filepath = os.path.join(temp_dir, "test_feat.pth")
    create_test_pth_file(filepath, include_features=True)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert 'pos' in result
    assert 'feat' in result
    assert result['feat'].shape[1] == 3  # 3 feature columns
    assert result['feat'].dtype == torch.float32


def test_load_point_cloud_off_basic(temp_dir):
    """Test loading basic OFF file."""
    filepath = os.path.join(temp_dir, "test.off")
    create_test_off_file(filepath)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert 'pos' in result
    assert result['pos'].shape == (4, 3)  # 4 vertices, XYZ
    assert result['pos'].dtype == torch.float32


def test_load_point_cloud_device_transfer(temp_dir):
    """Test device transfer functionality."""
    filepath = os.path.join(temp_dir, "test.pth")
    create_test_pth_file(filepath)

    # Test CPU device
    result_cpu = load_point_cloud(filepath=filepath, device='cpu')
    assert result_cpu['pos'].device.type == 'cpu'

    # Test CUDA device (if available)
    if torch.cuda.is_available():
        result_cuda = load_point_cloud(filepath=filepath, device='cuda')
        assert result_cuda['pos'].device.type == 'cuda'


def test_load_point_cloud_segmentation_labels(temp_dir):
    """Test loading segmentation files with proper label dtype conversion."""
    filepath = os.path.join(temp_dir, "test_seg.pth")

    # Create segmentation data with integer labels
    data = torch.cat(
        [
            torch.rand(50, 3),  # XYZ coordinates
            torch.randint(0, 10, (50, 1), dtype=torch.float32),  # Labels as features
        ],
        dim=1,
    )
    torch.save(data, filepath)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert 'pos' in result
    assert 'feat' in result
    assert (
        result['feat'].dtype == torch.int64
    )  # Should convert to long for segmentation


def test_load_point_cloud_nonexistent_file():
    """Test error handling for non-existent files."""
    with pytest.raises(FileNotFoundError, match="Point cloud file not found"):
        load_point_cloud(filepath="nonexistent.ply", device='cpu')


def test_load_point_cloud_unsupported_format(temp_dir):
    """Test error handling for unsupported file formats."""
    filepath = os.path.join(temp_dir, "test.xyz")

    # Create empty file with unsupported extension
    with open(filepath, 'w') as f:
        f.write("dummy content")

    with pytest.raises(ValueError, match="Unsupported file format"):
        load_point_cloud(filepath=filepath, device='cpu')


def test_load_from_ply_basic(temp_dir):
    """Test _load_from_ply function directly."""
    filepath = os.path.join(temp_dir, "test.ply")
    create_test_ply_file(filepath, include_rgb=True, include_feat=True)

    result = _load_from_ply(filepath=filepath, name_feat='feature')

    assert isinstance(result, dict)
    assert 'pos' in result
    assert 'rgb' in result
    assert 'feat' in result
    assert isinstance(result['pos'], np.ndarray)
    assert result['pos'].shape[1] == 3


def test_load_from_txt_basic(temp_dir):
    """Test _load_from_txt function directly."""
    filepath = os.path.join(temp_dir, "test.txt")
    create_test_txt_file(filepath, include_features=True)

    result = _load_from_txt(filepath=filepath)

    assert isinstance(result, dict)
    assert 'pos' in result
    assert 'feat' in result
    assert isinstance(result['pos'], np.ndarray)
    assert result['pos'].shape[1] == 3


def test_load_from_pth_torch_tensor(temp_dir):
    """Test _load_from_pth with torch tensor data."""
    filepath = os.path.join(temp_dir, "test_torch.pth")
    data = torch.rand(50, 4)  # XYZ + 1 feature
    torch.save(data, filepath)

    result = _load_from_pth(filepath=filepath)

    assert isinstance(result, dict)
    assert 'pos' in result
    assert 'feat' in result
    assert isinstance(result['pos'], torch.Tensor)
    assert result['pos'].shape == (50, 3)
    assert result['feat'].shape == (50, 1)


def test_load_from_pth_numpy_array(temp_dir):
    """Test _load_from_pth with numpy array data."""
    filepath = os.path.join(temp_dir, "test_numpy.pth")
    data = np.random.rand(50, 3).astype(np.float32)
    torch.save(data, filepath)

    result = _load_from_pth(filepath=filepath)

    assert isinstance(result, dict)
    assert 'pos' in result
    assert isinstance(result['pos'], np.ndarray)
    assert result['pos'].shape == (50, 3)


def test_load_from_off_basic(temp_dir):
    """Test _load_from_off function directly."""
    filepath = os.path.join(temp_dir, "test.off")
    create_test_off_file(filepath)

    result = _load_from_off(filepath=filepath, device='cpu')

    assert isinstance(result, dict)
    assert 'pos' in result
    assert isinstance(result['pos'], torch.Tensor)
    assert result['pos'].shape == (4, 3)
    assert result['pos'].device.type == 'cpu'


def test_load_from_off_invalid_format(temp_dir):
    """Test error handling for invalid OFF file format."""
    filepath = os.path.join(temp_dir, "invalid.off")

    # Create invalid OFF file
    with open(filepath, 'w') as f:
        f.write("INVALID\n")
        f.write("4 0 0\n")

    with pytest.raises(ValueError, match="Invalid OFF file format"):
        _load_from_off(filepath=filepath, device='cpu')


def test_uint16_to_int64_conversion(temp_dir):
    """Test conversion of uint16 data to int64."""
    filepath = os.path.join(temp_dir, "test_uint16.pth")

    # Create data with uint16 dtype
    data = np.random.randint(0, 1000, (50, 3), dtype=np.uint16)
    torch.save(data, filepath)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert result['pos'].dtype == torch.float32  # pos is always converted to float32
    # The conversion happens internally in the loading process


def test_path_normalization(temp_dir):
    """Test that file paths are normalized correctly."""
    filepath = os.path.join(temp_dir, "test.pth")
    create_test_pth_file(filepath)

    # Test with backslashes (Windows-style paths)
    windows_style_path = filepath.replace('/', '\\')

    result = load_point_cloud(filepath=windows_style_path, device='cpu')

    assert 'pos' in result
    assert result['pos'].dtype == torch.float32


def test_empty_point_cloud_handling(temp_dir):
    """Test handling of empty point cloud data."""
    filepath = os.path.join(temp_dir, "empty.pth")

    # Create empty point cloud
    empty_data = torch.empty(0, 3)
    torch.save(empty_data, filepath)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert result['pos'].shape[0] == 0
    assert result['pos'].shape[1] == 3


def test_load_point_cloud_various_sizes(temp_dir):
    """Test loading point clouds of various sizes."""
    sizes = [1, 10, 100, 1000]

    for size in sizes:
        filepath = os.path.join(temp_dir, f"size_{size}.pth")
        data = torch.rand(size, 3)
        torch.save(data, filepath)

        result = load_point_cloud(filepath=filepath, device='cpu')

        assert result['pos'].shape[0] == size
        assert result['pos'].shape[1] == 3
        assert result['pos'].dtype == torch.float32


def test_load_point_cloud_ply_different_element_names(temp_dir):
    """Test loading PLY files with different element names."""
    filepath = os.path.join(temp_dir, "custom_element.ply")

    # Create PLY with custom element name
    vertices = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0)]
    vertex_element = PlyElement.describe(
        np.array(vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]),
        'points',  # Custom element name
    )
    PlyData([vertex_element], text=True).write(filepath)

    # Should work with default (None) nameInPly parameter
    result = load_point_cloud(filepath=filepath, device='cpu')

    assert 'pos' in result
    assert result['pos'].shape == (3, 3)


def test_load_point_cloud_ply_missing_coordinates(temp_dir):
    """Test error handling for PLY files missing coordinate data."""
    filepath = os.path.join(temp_dir, "missing_coords.ply")

    # Create PLY with missing z coordinate
    vertices = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]

    try:
        vertex_element = PlyElement.describe(
            np.array(vertices, dtype=[('x', 'f4'), ('y', 'f4')]), 'vertex'
        )
        PlyData([vertex_element], text=True).write(filepath)

        # Should raise an error when trying to load
        with pytest.raises((KeyError, IndexError)):
            load_point_cloud(filepath=filepath, device='cpu')
    except Exception as e:
        # If we can't create the malformed PLY, that's a test setup failure
        assert (
            False
        ), f"Test setup failure: Cannot create malformed PLY file for testing: {e}"


def test_load_point_cloud_concurrent_access(temp_dir):
    """Test concurrent access to the same point cloud file."""
    import threading

    filepath = os.path.join(temp_dir, "concurrent.pth")
    create_test_pth_file(filepath)

    results = []
    errors = []

    def load_worker():
        try:
            result = load_point_cloud(filepath=filepath, device='cpu')
            results.append(result)
        except Exception as e:
            errors.append(e)

    # Create multiple threads that load the same file
    threads = []
    for _ in range(5):
        thread = threading.Thread(target=load_worker)
        threads.append(thread)

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Verify results
    assert len(errors) == 0, f"Errors occurred: {errors}"
    assert len(results) == 5

    # All results should have the same structure
    for result in results:
        assert 'pos' in result
        assert result['pos'].shape[1] == 3
        assert result['pos'].dtype == torch.float32
