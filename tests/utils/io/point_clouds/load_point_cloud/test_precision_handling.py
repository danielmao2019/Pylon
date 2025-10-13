import pytest
import tempfile
import numpy as np
import torch
from plyfile import PlyData, PlyElement

from utils.io.point_clouds.load_point_cloud import load_point_cloud


# ============================================================================
# PRECISION PRESERVATION TESTS - LARGE UTM COORDINATES
# ============================================================================


def test_float64_precision_preservation_large_utm_coordinates():
    """Test float64 preserves precision for large UTM coordinates like iVISION dataset."""
    # Real UTM coordinates from iVISION dataset (Zone 17N)
    large_coords = np.array(
        [
            [537000.123456, 4805000.654321, 350.987654],
            [537001.123456, 4805001.654321, 351.987654],
        ],
        dtype=np.float64,
    )

    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        # Create PLY with float64 precision
        vertex_data = np.array(
            [(x, y, z) for x, y, z in large_coords],
            dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8')],
        )

        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        PlyData([vertex_element]).write(tmp_file.name)

        # Load with float64 precision
        result = load_point_cloud(tmp_file.name, dtype=torch.float64)
        loaded_coords = result['pos'].cpu().numpy()

        # Check precision preservation
        np.testing.assert_allclose(
            loaded_coords,
            large_coords,
            rtol=1e-15,
            atol=1e-15,
            err_msg="float64 precision not preserved for large UTM coordinates",
        )


def test_float32_precision_loss_large_utm_coordinates():
    """Test that float32 loses precision for large UTM coordinates."""
    large_coords = np.array(
        [
            [537000.0001, 4805000.0001, 350.0001],  # Sub-millimeter precision
            [537000.0002, 4805000.0002, 350.0002],
        ],
        dtype=np.float64,
    )

    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        vertex_data = np.array(
            [(x, y, z) for x, y, z in large_coords],
            dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8')],
        )

        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        PlyData([vertex_element]).write(tmp_file.name)

        # Load with both precisions
        result_f32 = load_point_cloud(tmp_file.name, dtype=torch.float32)
        result_f64 = load_point_cloud(tmp_file.name, dtype=torch.float64)

        coords_f32 = result_f32['pos'].cpu().numpy()
        coords_f64 = result_f64['pos'].cpu().numpy()

        # float32 should NOT match original high precision
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(coords_f32, large_coords, rtol=1e-15, atol=1e-15)

        # But float64 should match
        np.testing.assert_allclose(coords_f64, large_coords, rtol=1e-15, atol=1e-15)


def test_millimeter_precision_preservation():
    """Test preservation of millimeter-level precision at UTM scale."""
    base_coord = np.array([[537123.456789, 4805678.123456, 345.678901]])

    # Add millimeter-level offsets
    mm_offsets = np.array(
        [
            [0.000, 0.000, 0.000],
            [0.001, 0.001, 0.001],  # 1mm
            [0.002, 0.002, 0.002],  # 2mm
            [0.005, 0.005, 0.005],  # 5mm
        ]
    )

    coords = base_coord + mm_offsets

    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        vertex_data = np.array(
            [(x, y, z) for x, y, z in coords],
            dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8')],
        )

        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        PlyData([vertex_element]).write(tmp_file.name)

        # Load with float64
        result = load_point_cloud(tmp_file.name, dtype=torch.float64)
        loaded_coords = result['pos'].cpu().numpy()

        # Check that millimeter differences are preserved
        for i in range(1, len(coords)):
            diff_original = coords[i] - coords[0]
            diff_loaded = loaded_coords[i] - loaded_coords[0]

            np.testing.assert_allclose(
                diff_original,
                diff_loaded,
                rtol=1e-12,
                atol=1e-12,
                err_msg=f"Millimeter precision lost for point {i}",
            )


def test_small_coordinates_precision_both_dtypes():
    """Test that both float32 and float64 preserve precision for small coordinates."""
    small_coords = np.array(
        [
            [0.0001, 0.0001, 0.0001],
            [0.001, 0.001, 0.001],
            [0.01, 0.01, 0.01],
            [0.1, 0.1, 0.1],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        vertex_data = np.array(
            [(x, y, z) for x, y, z in small_coords],
            dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8')],
        )

        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        PlyData([vertex_element]).write(tmp_file.name)

        # Load with both precisions
        result_f32 = load_point_cloud(tmp_file.name, dtype=torch.float32)
        result_f64 = load_point_cloud(tmp_file.name, dtype=torch.float64)

        coords_f32 = result_f32['pos'].cpu().numpy()
        coords_f64 = result_f64['pos'].cpu().numpy()

        # Both should preserve precision for small coordinates
        np.testing.assert_allclose(
            coords_f32,
            small_coords.astype(np.float32),
            rtol=1e-6,
            atol=1e-6,
            err_msg="float32 precision not preserved for small coordinates",
        )

        np.testing.assert_allclose(
            coords_f64,
            small_coords,
            rtol=1e-15,
            atol=1e-15,
            err_msg="float64 precision not preserved for small coordinates",
        )


# ============================================================================
# TXT FORMAT PRECISION TESTS
# ============================================================================


def test_txt_format_precision_preservation():
    """Test that TXT loading preserves precision when using float64."""
    large_coords = np.array(
        [
            [537000.123456, 4805000.654321, 350.987654],
            [537001.123456, 4805001.654321, 351.987654],
        ],
        dtype=np.float64,
    )

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
        tmp_file.write("# Point cloud data\n")
        tmp_file.write("# X Y Z\n")
        for x, y, z in large_coords:
            tmp_file.write(f"{x:.10f} {y:.10f} {z:.10f}\n")
        tmp_file.flush()

        # Load with float64 precision
        result = load_point_cloud(tmp_file.name, dtype=torch.float64)
        loaded_coords = result['pos'].cpu().numpy()

        # Check precision preservation (slightly relaxed due to text parsing)
        np.testing.assert_allclose(
            loaded_coords,
            large_coords,
            rtol=1e-10,
            atol=1e-10,
            err_msg="TXT format precision not preserved for large UTM coordinates",
        )


# ============================================================================
# DEVICE AND DTYPE HANDLING TESTS
# ============================================================================


def test_dtype_parameter_respected():
    """Test that the dtype parameter is correctly applied."""
    coords = np.array(
        [
            [537000.123456, 4805000.654321, 350.987654],
            [537001.123456, 4805001.654321, 351.987654],
        ],
        dtype=np.float64,
    )

    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        vertex_data = np.array(
            [(x, y, z) for x, y, z in coords],
            dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8')],
        )

        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        PlyData([vertex_element]).write(tmp_file.name)

        # Test float32 request
        result_f32 = load_point_cloud(tmp_file.name, dtype=torch.float32)
        assert result_f32['pos'].dtype == torch.float32

        # Test float64 request
        result_f64 = load_point_cloud(tmp_file.name, dtype=torch.float64)
        assert result_f64['pos'].dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_transfer_precision_preservation():
    """Test that device transfer doesn't affect precision."""
    coords = np.array(
        [
            [537000.123456, 4805000.654321, 350.987654],
            [537001.123456, 4805001.654321, 351.987654],
        ],
        dtype=np.float64,
    )

    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        vertex_data = np.array(
            [(x, y, z) for x, y, z in coords],
            dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8')],
        )

        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        PlyData([vertex_element]).write(tmp_file.name)

        # Test CPU and CUDA devices
        result_cpu = load_point_cloud(tmp_file.name, device='cpu', dtype=torch.float64)
        result_cuda = load_point_cloud(
            tmp_file.name, device='cuda', dtype=torch.float64
        )

        # Compare CPU and GPU results
        coords_cpu = result_cpu['pos'].cpu().numpy()
        coords_cuda = result_cuda['pos'].cpu().numpy()

        np.testing.assert_allclose(
            coords_cpu,
            coords_cuda,
            rtol=1e-15,
            atol=1e-15,
            err_msg="Device transfer affects precision",
        )


def test_internal_float64_processing():
    """Test that internal processing uses float64 even when output is float32."""
    coords = np.array(
        [
            [537000.123456, 4805000.654321, 350.987654],
            [537001.123456, 4805001.654321, 351.987654],
        ],
        dtype=np.float64,
    )

    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        vertex_data = np.array(
            [(x, y, z) for x, y, z in coords],
            dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8')],
        )

        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        PlyData([vertex_element]).write(tmp_file.name)

        # Load as float32 (but should use float64 internally)
        result_f32 = load_point_cloud(tmp_file.name, dtype=torch.float32)

        # Load as float64 for comparison
        result_f64 = load_point_cloud(tmp_file.name, dtype=torch.float64)

        # Convert f32 result back to f64 for comparison
        coords_f32_as_f64 = result_f32['pos'].double().cpu().numpy()
        coords_f64 = result_f64['pos'].cpu().numpy()

        # The difference should be due to final conversion, not internal processing
        max_error = np.max(np.abs(coords_f32_as_f64 - coords_f64))

        # Error should be consistent with float32 precision limits
        expected_error = np.max(np.abs(coords)) * np.finfo(np.float32).eps
        assert max_error <= expected_error * 10  # Allow some margin


# ============================================================================
# EDGE CASES AND EXTREME VALUES
# ============================================================================


def test_extreme_coordinate_values():
    """Test handling of extreme coordinate values."""
    extreme_coords = np.array(
        [
            [999999.999999, 9999999.999999, 9999.999999],
            [0.000001, 0.000001, 0.000001],
            [-999999.999999, -9999999.999999, -9999.999999],
        ],
        dtype=np.float64,
    )

    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        vertex_data = np.array(
            [(x, y, z) for x, y, z in extreme_coords],
            dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8')],
        )

        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        PlyData([vertex_element]).write(tmp_file.name)

        # Load with float64
        result = load_point_cloud(tmp_file.name, dtype=torch.float64)
        loaded_coords = result['pos'].cpu().numpy()

        # Check precision preservation for extreme values
        np.testing.assert_allclose(
            loaded_coords,
            extreme_coords,
            rtol=1e-14,
            atol=1e-14,
            err_msg="Extreme coordinate values lose precision",
        )


def test_zero_coordinate_precision():
    """Test that zero coordinates are handled correctly."""
    coords_with_zeros = np.array(
        [
            [0.0, 0.0, 0.0],
            [537000.000001, 0.0, 0.0],
            [0.0, 4805000.000001, 0.0],
            [0.0, 0.0, 350.000001],
        ],
        dtype=np.float64,
    )

    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        vertex_data = np.array(
            [(x, y, z) for x, y, z in coords_with_zeros],
            dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8')],
        )

        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        PlyData([vertex_element]).write(tmp_file.name)

        result = load_point_cloud(tmp_file.name, dtype=torch.float64)
        loaded_coords = result['pos'].cpu().numpy()

        # Exact zero should be preserved
        zero_mask = coords_with_zeros == 0.0
        assert np.all(
            loaded_coords[zero_mask] == 0.0
        ), "Zero coordinates not preserved exactly"

        # Non-zero small values should be preserved
        np.testing.assert_allclose(
            loaded_coords,
            coords_with_zeros,
            rtol=1e-15,
            atol=1e-15,
            err_msg="Small non-zero coordinates near zero lose precision",
        )


def test_coordinate_system_transformation_precision():
    """Test precision requirements for coordinate system transformations."""
    utm_coords = np.array(
        [
            [537000.123456, 4805000.654321, 350.987654],
            [537001.123456, 4805001.654321, 351.987654],
        ],
        dtype=np.float64,
    )

    # Small translation that should be preserved exactly
    translation = np.array([0.001, 0.001, 0.001])  # 1mm translation

    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        vertex_data = np.array(
            [(x, y, z) for x, y, z in utm_coords],
            dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8')],
        )

        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        PlyData([vertex_element]).write(tmp_file.name)

        # Load coordinates
        result = load_point_cloud(tmp_file.name, dtype=torch.float64)
        original_coords = result['pos'].cpu().numpy()

        # Apply small transformation
        transformed_coords = original_coords + translation

        # Check that transformation preserves precision
        computed_translation = transformed_coords - original_coords

        # Check transformation is preserved within reasonable float64 precision
        np.testing.assert_allclose(
            computed_translation,
            np.tile(translation, (len(utm_coords), 1)),
            rtol=1e-9,
            atol=2e-10,
            err_msg="Small coordinate transformations lose significant precision",
        )
