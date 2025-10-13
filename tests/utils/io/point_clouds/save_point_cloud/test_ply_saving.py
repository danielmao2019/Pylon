import pytest
import tempfile
import numpy as np
import torch
from plyfile import PlyData

from utils.io.point_clouds.save_point_cloud import save_point_cloud
from utils.io.point_clouds.load_point_cloud import load_point_cloud


# ============================================================================
# BASIC PLY SAVING TESTS
# ============================================================================


def test_basic_ply_saving():
    """Test basic PLY file saving and round-trip verification."""
    pc_data = {
        'pos': torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
            ],
            dtype=torch.float32,
        )
    }

    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        save_point_cloud(pc_data, tmp_file.name)

        # Verify file was created
        assert tmp_file.name.endswith('.ply')

        # Load back and verify content
        loaded = load_point_cloud(tmp_file.name)
        np.testing.assert_allclose(
            loaded['pos'].cpu().numpy(), pc_data['pos'].cpu().numpy(), rtol=1e-6
        )


def test_numpy_array_input():
    """Test saving with numpy array input."""
    pc_data = {
        'pos': np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
            ],
            dtype=np.float64,
        )
    }

    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        save_point_cloud(pc_data, tmp_file.name)

        # Load back and verify content
        loaded = load_point_cloud(tmp_file.name)
        np.testing.assert_allclose(
            loaded['pos'].cpu().numpy(), pc_data['pos'].astype(np.float32), rtol=1e-6
        )


def test_large_coordinates_precision():
    """Test saving large coordinates with proper precision handling."""
    # UTM coordinates similar to iVISION dataset
    large_coords = torch.tensor(
        [
            [537000.123456, 4805000.654321, 350.987654],
            [537001.123456, 4805001.654321, 351.987654],
        ],
        dtype=torch.float64,
    )

    pc_data = {'pos': large_coords}

    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        save_point_cloud(pc_data, tmp_file.name)

        # Load back and verify precision is maintained within float32 limits
        loaded = load_point_cloud(tmp_file.name, dtype=torch.float64)

        # Check that the precision loss is within expected float32 range
        max_error = torch.max(
            torch.abs(loaded['pos'].cpu() - large_coords.float().cpu())
        )
        expected_error = (
            torch.max(torch.abs(large_coords)) * torch.finfo(torch.float32).eps
        )

        # Allow some margin for PLY format conversion
        assert max_error <= expected_error * 100


def test_empty_point_cloud():
    """Test saving empty point cloud."""
    pc_data = {'pos': torch.empty((0, 3), dtype=torch.float32)}

    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        save_point_cloud(pc_data, tmp_file.name)

        # Should create valid PLY file with 0 vertices
        ply_data = PlyData.read(tmp_file.name)
        assert ply_data.elements[0].count == 0


# ============================================================================
# COLOR SAVING TESTS
# ============================================================================


def test_rgb_colors_saving():
    """Test saving with RGB colors."""
    pc_data = {
        'pos': torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
            ],
            dtype=torch.float32,
        ),
        'rgb': torch.tensor(
            [
                [1.0, 0.0, 0.0],  # Red
                [0.0, 1.0, 0.0],  # Green
                [0.0, 0.0, 1.0],  # Blue
            ],
            dtype=torch.float32,
        ),
    }

    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        save_point_cloud(pc_data, tmp_file.name)

        # Load back and verify colors
        loaded = load_point_cloud(tmp_file.name)
        assert 'rgb' in loaded

        # RGB should be preserved within uint8 quantization tolerance
        np.testing.assert_allclose(
            loaded['rgb'].cpu().numpy(),
            pc_data['rgb'].cpu().numpy(),
            rtol=1e-2,  # Allow for uint8 quantization error
        )


def test_colors_field_mapping():
    """Test that 'colors' field is mapped to RGB."""
    pc_data = {
        'pos': torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        'colors': torch.tensor(
            [
                [0.5, 0.5, 0.5],
                [0.8, 0.2, 0.1],
            ],
            dtype=torch.float32,
        ),
    }

    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        save_point_cloud(pc_data, tmp_file.name)

        # Load back and verify colors are available as 'rgb'
        loaded = load_point_cloud(tmp_file.name)
        assert 'rgb' in loaded

        # Colors should be preserved within uint8 quantization tolerance
        np.testing.assert_allclose(
            loaded['rgb'].cpu().numpy(),
            pc_data['colors'].cpu().numpy(),
            rtol=2e-2,  # Allow for uint8 quantization error
        )


def test_normalized_colors_conversion():
    """Test conversion of normalized [0,1] colors to [0,255] range."""
    pc_data = {
        'pos': torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        'rgb': torch.tensor([[0.5, 0.25, 0.75]], dtype=torch.float32),
    }

    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        save_point_cloud(pc_data, tmp_file.name)

        # Manually check PLY file has correct uint8 values
        ply_data = PlyData.read(tmp_file.name)
        vertex_data = ply_data.elements[0].data

        # Should be converted to uint8 range
        assert abs(vertex_data['red'][0] - 127) <= 1  # 0.5 * 255 ≈ 127
        assert abs(vertex_data['green'][0] - 63) <= 1  # 0.25 * 255 ≈ 63
        assert abs(vertex_data['blue'][0] - 191) <= 1  # 0.75 * 255 ≈ 191


# ============================================================================
# FEATURE SAVING TESTS
# ============================================================================


def test_single_feature_saving():
    """Test saving with a single feature column."""
    pc_data = {
        'pos': torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
            ],
            dtype=torch.float32,
        ),
        'intensity': torch.tensor(
            [
                [1.5],
                [2.5],
                [3.5],
            ],
            dtype=torch.float32,
        ),
    }

    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        save_point_cloud(pc_data, tmp_file.name)

        # Manually verify PLY file structure
        ply_data = PlyData.read(tmp_file.name)
        vertex_data = ply_data.elements[0].data

        # Should have intensity field
        assert 'intensity' in vertex_data.dtype.names
        np.testing.assert_allclose(vertex_data['intensity'], [1.5, 2.5, 3.5], rtol=1e-6)


def test_multiple_features_saving():
    """Test saving with multiple feature columns."""
    pc_data = {
        'pos': torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        'features': torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=torch.float32,
        ),
    }

    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        save_point_cloud(pc_data, tmp_file.name)

        # Manually verify PLY structure
        ply_data = PlyData.read(tmp_file.name)
        vertex_data = ply_data.elements[0].data

        # Should have feature columns
        for i in range(3):
            field_name = f'features_{i}'
            assert field_name in vertex_data.dtype.names


def test_none_field_handling():
    """Test that None fields are skipped gracefully."""
    pc_data = {
        'pos': torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        'colors': None,
        'features': torch.tensor([[1.0]], dtype=torch.float32),
    }

    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        save_point_cloud(pc_data, tmp_file.name)

        # Should save successfully, skipping None field
        ply_data = PlyData.read(tmp_file.name)
        vertex_data = ply_data.elements[0].data

        # Should have pos and features, but not colors
        assert 'x' in vertex_data.dtype.names
        assert 'features' in vertex_data.dtype.names
        assert 'red' not in vertex_data.dtype.names


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


def test_missing_pos_field_error():
    """Test error when 'pos' field is missing."""
    pc_data = {'colors': torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)}

    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        with pytest.raises(
            AssertionError, match="Point cloud data must contain 'pos' field"
        ):
            save_point_cloud(pc_data, tmp_file.name)


def test_wrong_file_extension_error():
    """Test error when file extension is not .ply."""
    pc_data = {'pos': torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)}

    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
        with pytest.raises(ValueError, match="Unsupported output format"):
            save_point_cloud(pc_data, tmp_file.name)


def test_invalid_pos_shape_error():
    """Test error when positions have wrong shape."""
    pc_data = {'pos': torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)}  # Wrong shape

    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        with pytest.raises(AssertionError, match="Expected positions shape \\(N, 3\\)"):
            save_point_cloud(pc_data, tmp_file.name)


# ============================================================================
# DEVICE HANDLING TESTS
# ============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_tensor_saving():
    """Test saving tensors from CUDA device."""
    pc_data = {
        'pos': torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
            device='cuda',
        )
    }

    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        save_point_cloud(pc_data, tmp_file.name)

        # Should save successfully and load back correctly
        loaded = load_point_cloud(tmp_file.name)
        np.testing.assert_allclose(
            loaded['pos'].cpu().numpy(), pc_data['pos'].cpu().numpy(), rtol=1e-6
        )


def test_mixed_tensor_types_saving():
    """Test saving with mixed numpy arrays and torch tensors."""
    pc_data = {
        'pos': np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        ),
        'colors': torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    }

    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        save_point_cloud(pc_data, tmp_file.name)

        # Should save successfully
        loaded = load_point_cloud(tmp_file.name)
        assert 'pos' in loaded
        assert 'rgb' in loaded


# ============================================================================
# INTEGRATION TESTS WITH LOAD FUNCTIONALITY
# ============================================================================


@pytest.mark.parametrize(
    "coordinates,features",
    [
        # Basic coordinates only
        (np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), None),
        # Large UTM coordinates
        (np.array([[537000.1, 4805000.2, 350.3]]), None),
        # With features
        (np.array([[0.0, 0.0, 0.0]]), np.array([[1.5]])),
    ],
)
def test_save_load_round_trip(coordinates, features):
    """Test complete save-load round trip preserves data."""
    pc_data = {'pos': torch.from_numpy(coordinates).float()}

    if features is not None:
        pc_data['features'] = torch.from_numpy(features).float()

    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        save_point_cloud(pc_data, tmp_file.name)
        loaded = load_point_cloud(tmp_file.name)

        # Verify positions
        np.testing.assert_allclose(
            loaded['pos'].cpu().numpy(), coordinates.astype(np.float32), rtol=1e-6
        )

        # Verify features if present
        if features is not None:
            # Features may be saved with different field names
            feature_fields = [k for k in loaded.keys() if k.startswith('features')]
            assert len(feature_fields) > 0, "Features should be preserved"


def test_precision_consistency_save_load():
    """Test that precision is consistent between save and load operations."""
    # Test with coordinates that stress float32 precision
    coords = torch.tensor(
        [
            [537000.123, 4805000.456, 350.789],  # Large values
            [0.000123, 0.000456, 0.000789],  # Small values
        ],
        dtype=torch.float64,
    )

    pc_data = {'pos': coords}

    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        save_point_cloud(pc_data, tmp_file.name)

        # Load with same precision as save
        loaded_f32 = load_point_cloud(tmp_file.name, dtype=torch.float32)
        loaded_f64 = load_point_cloud(tmp_file.name, dtype=torch.float64)

        # Both should be consistent with expected float32 precision limits
        coords_f32 = coords.float()

        np.testing.assert_allclose(
            loaded_f32['pos'].cpu().numpy(), coords_f32.cpu().numpy(), rtol=1e-6
        )

        # float64 should be limited by the float32 storage format
        max_error = torch.max(torch.abs(loaded_f64['pos'].cpu() - coords_f32.cpu()))
        expected_error = torch.max(torch.abs(coords)) * torch.finfo(torch.float32).eps
        assert max_error <= expected_error * 10
