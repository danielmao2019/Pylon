import os
import tempfile
import numpy as np
import torch
from PIL import Image
import pytest
from utils.io.image import (
    load_image,
    _load_image,
    _load_multispectral_image,
    _normalize,
)


def test_load_image_single_rgb():
    """Test loading a single RGB image file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "test_rgb.png")

        # Create test RGB image
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(img_array).save(filepath)

        # Load image
        result = load_image(filepath=filepath)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 100, 100)  # (C, H, W)
        assert result.dtype == torch.uint8


def test_load_image_single_grayscale():
    """Test loading a single grayscale image file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "test_gray.png")

        # Create test grayscale image
        img_array = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        Image.fromarray(img_array, mode='L').save(filepath)

        # Load image
        result = load_image(filepath=filepath)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (100, 100)  # (H, W)
        assert result.dtype == torch.uint8


def test_load_image_with_dtype_conversion():
    """Test loading image with dtype conversion."""
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "test_rgb.png")

        # Create test RGB image
        img_array = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        Image.fromarray(img_array).save(filepath)

        # Load with float32 conversion
        result = load_image(filepath=filepath, dtype=torch.float32)

        assert result.dtype == torch.float32
        assert result.shape == (3, 50, 50)


def test_load_image_with_sub_div_normalization():
    """Test loading image with subtraction and division normalization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "test_rgb.png")

        # Create test RGB image
        img_array = np.full((50, 50, 3), 100, dtype=np.uint8)
        Image.fromarray(img_array).save(filepath)

        # Load with sub/div normalization
        result = load_image(filepath=filepath, sub=50.0, div=2.0)

        assert result.dtype == torch.float32  # Normalization converts to float32
        expected_value = (100.0 - 50.0) / 2.0  # (100 - 50) / 2 = 25
        assert torch.allclose(
            result, torch.full((3, 50, 50), expected_value), atol=1e-5
        )


def test_load_image_with_min_max_normalization():
    """Test loading image with min-max normalization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "test_rgb.png")

        # Create test RGB image with known values
        img_array = np.array([[[0, 127, 255]]], dtype=np.uint8)  # 1x1x3 image
        Image.fromarray(img_array).save(filepath)

        # Load with min-max normalization
        result = load_image(filepath=filepath, normalization="min-max")

        assert result.dtype == torch.float32
        # Should normalize to [0, 1] range
        assert result.min() >= 0.0
        assert result.max() <= 1.0


def test_load_image_with_mean_std_normalization():
    """Test loading image with mean-std normalization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "test_rgb.png")

        # Create test RGB image
        img_array = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        Image.fromarray(img_array).save(filepath)

        # Load with mean-std normalization
        result = load_image(filepath=filepath, normalization="mean-std")

        assert result.dtype == torch.float32
        # Check that mean is approximately 0 and std is approximately 1
        assert abs(result.mean().item()) < 1e-5
        assert abs(result.std().item() - 1.0) < 1e-5


def test_load_image_invalid_inputs():
    """Test error handling for invalid inputs."""
    # Both filepath and filepaths provided
    with pytest.raises(
        ValueError, match="Exactly one of 'filepath' or 'filepaths' must be provided"
    ):
        load_image(filepath="test.png", filepaths=["test1.png", "test2.png"])

    # Neither provided
    with pytest.raises(
        ValueError, match="Exactly one of 'filepath' or 'filepaths' must be provided"
    ):
        load_image()

    # Invalid dtype - create a temp file first to avoid file not found error
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "test.png")
        img_array = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        Image.fromarray(img_array).save(filepath)

        with pytest.raises(TypeError, match="'dtype' must be a torch.dtype"):
            load_image(filepath=filepath, dtype="float32")

        # Normalization with sub/div
        with pytest.raises(
            ValueError,
            match="'normalization' cannot be used together with 'sub' or 'div'",
        ):
            load_image(filepath=filepath, normalization="min-max", sub=10.0)


def test_load_image_nonexistent_file():
    """Test error handling for non-existent files."""
    with pytest.raises((FileNotFoundError, AssertionError)):
        load_image(filepath="nonexistent.png")


def test_normalize_min_max():
    """Test min-max normalization function."""
    tensor = torch.tensor([[10.0, 20.0], [30.0, 40.0]])

    result = _normalize(tensor, sub=None, div=None, normalization="min-max")

    # Should normalize to [0, 1] range (with small tolerance for floating point precision)
    assert abs(result.min().item() - 0.0) < 1e-6
    assert abs(result.max().item() - 1.0) < 1e-6
    assert result.dtype == torch.float32


def test_normalize_mean_std():
    """Test mean-std normalization function."""
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    result = _normalize(tensor, sub=None, div=None, normalization="mean-std")

    # Should have mean ~0 and std ~1
    assert abs(result.mean().item()) < 1e-5
    assert abs(result.std().item() - 1.0) < 1e-5
    assert result.dtype == torch.float32


def test_normalize_sub_div():
    """Test subtraction and division normalization."""
    tensor = torch.tensor([[10.0, 20.0], [30.0, 40.0]])

    result = _normalize(tensor, sub=5.0, div=2.0, normalization=None)

    expected = (tensor - 5.0) / 2.0
    assert torch.allclose(result, expected)
    assert result.dtype == torch.float32


def test_normalize_no_operation():
    """Test normalization with no operations."""
    tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8)

    result = _normalize(tensor, sub=None, div=None, normalization=None)

    # Should return unchanged
    assert torch.equal(result, tensor)
    assert result.dtype == tensor.dtype


def test_normalize_division_by_zero_protection():
    """Test protection against division by zero."""
    tensor = torch.zeros(2, 2)

    # Test div parameter protection
    result = _normalize(tensor, sub=None, div=0.0, normalization=None)
    assert torch.all(torch.isfinite(result))

    # Test mean-std protection (zero std)
    result = _normalize(tensor, sub=None, div=None, normalization="mean-std")
    assert torch.all(torch.isfinite(result))

    # Test min-max protection (zero range)
    result = _normalize(tensor, sub=None, div=None, normalization="min-max")
    assert torch.all(torch.isfinite(result))


def test_normalize_multichannel():
    """Test normalization with multi-channel tensors."""
    # 3-channel tensor (C, H, W)
    tensor = torch.rand(3, 10, 10)

    # Test with per-channel sub/div
    sub_per_channel = [0.1, 0.2, 0.3]
    div_per_channel = [2.0, 3.0, 4.0]

    result = _normalize(
        tensor, sub=sub_per_channel, div=div_per_channel, normalization=None
    )

    assert result.shape == tensor.shape
    assert result.dtype == torch.float32

    # Verify per-channel normalization was applied correctly
    for c in range(3):
        expected_channel = (tensor[c] - sub_per_channel[c]) / div_per_channel[c]
        assert torch.allclose(result[c], expected_channel, atol=1e-6)


def test_normalize_invalid_inputs():
    """Test error handling for invalid normalization inputs."""
    tensor = torch.rand(3, 10, 10)

    # Invalid normalization method
    with pytest.raises(AssertionError):
        _normalize(tensor, sub=None, div=None, normalization="invalid")

    # Conflicting normalization and sub/div
    with pytest.raises(
        ValueError, match="'normalization' cannot be used together with 'sub' or 'div'"
    ):
        _normalize(tensor, sub=10.0, div=None, normalization="min-max")

    # Wrong number of values for multi-channel
    with pytest.raises(
        ValueError, match="Normalization value must match the number of channels"
    ):
        _normalize(
            tensor, sub=[0.1, 0.2], div=None, normalization=None
        )  # 2 values for 3 channels
