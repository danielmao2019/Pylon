"""Integration tests for all classic point cloud registration methods."""
import pytest
import torch
import time
from tests.models.utils.point_cloud_data import generate_point_cloud_data
from .common import validate_transformation_output


@pytest.mark.parametrize("model_config", [
    ("ICP", "models.point_cloud_registration.classic.icp", "ICP", {"threshold": 0.02, "max_iterations": 20}),
    ("RANSAC_FPFH", "models.point_cloud_registration.classic.ransac_fpfh", "RANSAC_FPFH", {"voxel_size": 0.05}),
    ("TeaserPlusPlus", "models.point_cloud_registration.classic.teaserplusplus", "TeaserPlusPlus", {"voxel_size": 0.05}),
])
def test_classic_methods_basic_functionality(model_config):
    """Test basic functionality of all classic registration methods."""
    model_name, module_path, class_name, init_args = model_config

    # Dynamic import
    module = __import__(module_path, fromlist=[class_name])
    model_class = getattr(module, class_name)
    model = model_class(**init_args)

    # Basic functionality test
    dummy_input = generate_point_cloud_data(
        batch_size=1, num_points=100, feature_dim=32
    )

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    # Validate transformation output
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 4, 4)

    # Check if it's a valid transformation matrix
    T = output[0]
    expected_bottom = torch.tensor([0., 0., 0., 1.])
    bottom_row = T[3, :]
    assert torch.allclose(bottom_row, expected_bottom, atol=1e-5)


def test_classic_methods_computational_efficiency():
    """Test that classic methods complete in reasonable time."""
    model_configs = [
        ("models.point_cloud_registration.classic.icp", "ICP", {"threshold": 0.02, "max_iterations": 10}),
        ("models.point_cloud_registration.classic.ransac_fpfh", "RANSAC_FPFH", {"voxel_size": 0.1}),
    ]

    for module_path, class_name, init_args in model_configs:
        # Dynamic import
        module = __import__(module_path, fromlist=[class_name])
        model_class = getattr(module, class_name)
        model = model_class(**init_args)

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
        assert computation_time < 5.0, f"{class_name} took too long: {computation_time:.2f}s"
        assert output is not None


@pytest.mark.parametrize("method_name,module_path,class_name", [
    ("ICP", "models.point_cloud_registration.classic.icp", "ICP"),
    ("RANSAC_FPFH", "models.point_cloud_registration.classic.ransac_fpfh", "RANSAC_FPFH"),
    ("TeaserPlusPlus", "models.point_cloud_registration.classic.teaserplusplus", "TeaserPlusPlus"),
])
def test_classic_methods_batch_consistency(method_name, module_path, class_name):
    """Test that all methods handle different batch sizes consistently."""
    # Dynamic import
    module = __import__(module_path, fromlist=[class_name])
    model_class = getattr(module, class_name)

    # Use minimal parameters for faster testing
    if class_name == "ICP":
        model = model_class(threshold=0.02, max_iterations=10)
    elif class_name == "RANSAC_FPFH":
        model = model_class(voxel_size=0.1)
    else:  # TeaserPlusPlus
        model = model_class(voxel_size=0.1)

    batch_sizes = [1, 2, 3]

    for batch_size in batch_sizes:
        dummy_input = generate_point_cloud_data(
            batch_size=batch_size, num_points=64, feature_dim=32
        )

        model.eval()
        with torch.no_grad():
            output = model(dummy_input)

        validate_transformation_output(output, batch_size=batch_size)


def test_classic_methods_device_consistency():
    """Test that methods work consistently across CPU/GPU."""
    model_configs = [
        ("models.point_cloud_registration.classic.icp", "ICP", {"threshold": 0.02, "max_iterations": 5}),
    ]

    for module_path, class_name, init_args in model_configs:
        # Dynamic import
        module = __import__(module_path, fromlist=[class_name])
        model_class = getattr(module, class_name)
        model = model_class(**init_args)

        # Test on CPU
        dummy_input_cpu = generate_point_cloud_data(
            batch_size=1, num_points=32, feature_dim=32, device='cpu'
        )

        model.eval()
        with torch.no_grad():
            output_cpu = model(dummy_input_cpu)

        validate_transformation_output(output_cpu, batch_size=1)
        assert output_cpu.device.type == 'cpu'

        # Test on GPU if available
        if torch.cuda.is_available():
            model_gpu = model.cuda()
            dummy_input_gpu = {
                k: {kk: vv.cuda() for kk, vv in v.items()}
                for k, v in dummy_input_cpu.items()
            }

            with torch.no_grad():
                output_gpu = model_gpu(dummy_input_gpu)

            validate_transformation_output(output_gpu, batch_size=1)
            assert output_gpu.device.type == 'cuda'
