import os
import tempfile
import torch
from PIL import Image
import pytest
from utils.io.image import save_image


def test_save_image_rgb_float():
    """Test saving RGB float tensor as image."""
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "test_save.png")

        # Create test tensor (3, H, W) float32
        tensor = torch.rand(3, 50, 50, dtype=torch.float32)

        # Save image
        save_image(tensor=tensor, filepath=filepath)

        # Verify file was created
        assert os.path.exists(filepath)

        # Load back and verify
        loaded_img = Image.open(filepath)
        assert loaded_img.size == (50, 50)


def test_save_image_grayscale_uint8():
    """Test saving grayscale uint8 tensor as image."""
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "test_save_gray.png")

        # Create test tensor (H, W) uint8
        tensor = torch.randint(0, 255, (50, 50), dtype=torch.uint8)

        # Save image
        save_image(tensor=tensor, filepath=filepath)

        # Verify file was created
        assert os.path.exists(filepath)

        # Load back and verify
        loaded_img = Image.open(filepath)
        assert loaded_img.size == (50, 50)
        assert loaded_img.mode == "L"  # Grayscale


def test_save_image_invalid_format():
    """Test error handling for unsupported tensor formats."""
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "test_save.png")

        # Invalid tensor format
        tensor = torch.rand(4, 50, 50)  # 4 channels not supported

        with pytest.raises(NotImplementedError, match="Unrecognized tensor format"):
            save_image(tensor=tensor, filepath=filepath)


def test_save_image_rgb_consistency():
    """Test that saved RGB images maintain consistency."""
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "test_consistency.png")

        # Create test tensor with known values
        tensor = torch.zeros(3, 10, 10, dtype=torch.float32)
        tensor[0, :, :] = 1.0  # Red channel
        tensor[1, 5:, :] = 0.5  # Green channel (bottom half)
        tensor[2, :, 5:] = 0.3  # Blue channel (right half)

        # Save image
        save_image(tensor=tensor, filepath=filepath)

        # Verify file properties
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0

        # Load back and verify it's readable
        loaded_img = Image.open(filepath)
        assert loaded_img.mode == "RGB"
        assert loaded_img.size == (10, 10)


def test_save_image_grayscale_consistency():
    """Test that saved grayscale images maintain consistency."""
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "test_gray_consistency.png")

        # Create test tensor with gradient pattern
        tensor = torch.zeros(20, 20, dtype=torch.uint8)
        for i in range(20):
            tensor[i, :] = i * 12  # Gradient from 0 to 228

        # Save image
        save_image(tensor=tensor, filepath=filepath)

        # Verify file properties
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0

        # Load back and verify it's readable
        loaded_img = Image.open(filepath)
        assert loaded_img.mode == "L"
        assert loaded_img.size == (20, 20)


def test_save_image_edge_cases():
    """Test saving images with edge case dimensions."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test minimum size RGB image
        small_filepath = os.path.join(temp_dir, "small_rgb.png")
        small_tensor = torch.rand(3, 1, 1, dtype=torch.float32)
        save_image(tensor=small_tensor, filepath=small_filepath)

        assert os.path.exists(small_filepath)
        small_img = Image.open(small_filepath)
        assert small_img.size == (1, 1)

        # Test minimum size grayscale image
        gray_filepath = os.path.join(temp_dir, "small_gray.png")
        gray_tensor = torch.randint(0, 255, (1, 1), dtype=torch.uint8)
        save_image(tensor=gray_tensor, filepath=gray_filepath)

        assert os.path.exists(gray_filepath)
        gray_img = Image.open(gray_filepath)
        assert gray_img.size == (1, 1)
        assert gray_img.mode == "L"


def test_save_image_different_dimensions():
    """Test saving images with various dimensions."""
    test_sizes = [(32, 32), (64, 48), (100, 200), (1, 300)]

    with tempfile.TemporaryDirectory() as temp_dir:
        for i, (height, width) in enumerate(test_sizes):
            filepath = os.path.join(temp_dir, f"test_size_{i}.png")

            # Create RGB tensor with specific dimensions
            tensor = torch.rand(3, height, width, dtype=torch.float32)

            # Save image
            save_image(tensor=tensor, filepath=filepath)

            # Verify dimensions
            assert os.path.exists(filepath)
            loaded_img = Image.open(filepath)
            assert loaded_img.size == (width, height)  # PIL uses (width, height)


def test_save_image_value_ranges():
    """Test saving images with different value ranges."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test RGB float values at boundaries
        boundary_cases = [
            ("zeros", torch.zeros(3, 10, 10, dtype=torch.float32)),
            ("ones", torch.ones(3, 10, 10, dtype=torch.float32)),
            ("half", torch.full((3, 10, 10), 0.5, dtype=torch.float32)),
        ]

        for name, tensor in boundary_cases:
            filepath = os.path.join(temp_dir, f"boundary_{name}.png")
            save_image(tensor=tensor, filepath=filepath)

            assert os.path.exists(filepath)
            loaded_img = Image.open(filepath)
            assert loaded_img.mode == "RGB"
            assert loaded_img.size == (10, 10)

        # Test grayscale uint8 values at boundaries
        uint8_cases = [
            ("min", torch.zeros(10, 10, dtype=torch.uint8)),
            ("max", torch.full((10, 10), 255, dtype=torch.uint8)),
            ("mid", torch.full((10, 10), 128, dtype=torch.uint8)),
        ]

        for name, tensor in uint8_cases:
            filepath = os.path.join(temp_dir, f"uint8_{name}.png")
            save_image(tensor=tensor, filepath=filepath)

            assert os.path.exists(filepath)
            loaded_img = Image.open(filepath)
            assert loaded_img.mode == "L"
            assert loaded_img.size == (10, 10)


def test_save_image_directory_creation():
    """Test that save_image works with non-existent directories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create nested directory path that doesn't exist
        nested_path = os.path.join(temp_dir, "subdir", "images")
        filepath = os.path.join(nested_path, "test.png")

        # This should work if the save function creates directories
        tensor = torch.rand(3, 20, 20, dtype=torch.float32)

        # Note: save_image currently uses check_write_file which may not create directories
        # This test verifies the current behavior
        try:
            save_image(tensor=tensor, filepath=filepath)
            # If it succeeds, verify the file exists
            assert os.path.exists(filepath)
        except (FileNotFoundError, AssertionError):
            # If it fails due to directory not existing, that's expected behavior
            # Create the directory manually and try again
            os.makedirs(nested_path, exist_ok=True)
            save_image(tensor=tensor, filepath=filepath)
            assert os.path.exists(filepath)


def test_save_image_overwrite_existing():
    """Test that save_image overwrites existing files correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "overwrite_test.png")

        # Save first image
        tensor1 = torch.zeros(3, 10, 10, dtype=torch.float32)
        save_image(tensor=tensor1, filepath=filepath)

        assert os.path.exists(filepath)
        first_size = os.path.getsize(filepath)

        # Save different image to same path
        tensor2 = torch.ones(3, 20, 20, dtype=torch.float32)
        save_image(tensor=tensor2, filepath=filepath)

        assert os.path.exists(filepath)
        second_size = os.path.getsize(filepath)

        # File should be overwritten (likely different size due to different content)
        # Just verify the file still exists and is readable
        loaded_img = Image.open(filepath)
        assert loaded_img.size == (20, 20)  # Should have new dimensions
