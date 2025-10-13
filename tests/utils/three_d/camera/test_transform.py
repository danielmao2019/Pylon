import torch
import pytest

from utils.three_d.camera.transform import world_to_camera_transform


@pytest.mark.parametrize(
    "points,extrinsics,expected",
    [
        (
            torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64),
            # Identity rotation, translation t = [1,2,3] in camera-to-world
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 2.0],
                    [0.0, 0.0, 1.0, 3.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=torch.float64,
            ),
            # world_to_camera = translate by -t, so point [1,2,3] -> [0,0,0]
            torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
        ),
    ],
)
def test_world_to_camera_numeric_correctness(points, extrinsics, expected):
    original = points.clone()
    out = world_to_camera_transform(points, extrinsics, inplace=False)
    assert torch.allclose(out, expected)
    assert out.data_ptr() != points.data_ptr()
    assert torch.allclose(points, original)


@pytest.mark.parametrize(
    "points,extrinsics,expected",
    [
        (
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 2.0]], dtype=torch.float32),
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.5],
                    [0.0, 1.0, 0.0, -0.5],
                    [0.0, 0.0, 1.0, 1.5],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            ),
            # Expected (world_to_camera with R=I): y = x - t
            # t = [0.5, -0.5, 1.5]
            # [1,0,0] - t = [0.5, 0.5, -1.5]
            # [0,1,2] - t = [-0.5, 1.5, 0.5]
            torch.tensor([[0.5, 0.5, -1.5], [-0.5, 1.5, 0.5]], dtype=torch.float32),
        ),
    ],
)
def test_world_to_camera_inplace_behavior(points, extrinsics, expected):
    out = world_to_camera_transform(points, extrinsics, inplace=True)
    assert torch.allclose(out, expected)
    assert out.data_ptr() == points.data_ptr(), "Expected in-place behavior"


@pytest.mark.parametrize("num_divide", [1, 2, 3])
def test_world_to_camera_num_divide_equivalence(num_divide):
    assert torch.cuda.is_available(), "CUDA is required for this test"
    device = torch.device('cuda:0')
    # Large point cloud to make batching meaningful but keep runtime manageable
    N = 1_000_000
    points = torch.randn((N, 3), dtype=torch.float32, device=device)
    extrinsics = torch.eye(4, dtype=torch.float32, device=device)
    extrinsics[:3, 3] = torch.tensor([0.5, -0.25, 1.0], dtype=torch.float32, device=device)

    # Baseline with no division
    baseline = world_to_camera_transform(points.clone(), extrinsics, inplace=False, num_divide=0)

    # Variant with division (out-of-place)
    out_oop = world_to_camera_transform(points.clone(), extrinsics, inplace=False, num_divide=num_divide)
    assert torch.equal(out_oop, baseline)

    # Variant with division (in-place)
    out_ip = world_to_camera_transform(points.clone(), extrinsics, inplace=True, num_divide=num_divide)
    assert torch.equal(out_ip, baseline)
