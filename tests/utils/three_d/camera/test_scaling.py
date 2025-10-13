import torch
import pytest

from utils.three_d.camera.scaling import scale_intrinsics


@pytest.mark.parametrize(
    "intrinsics,resolution,expected",
    [
        (
            torch.tensor(
                [[100.0, 0.0, 50.0], [0.0, 120.0, 100.0], [0.0, 0.0, 1.0]],
                dtype=torch.float64,
            ),
            (200, 400),
            torch.tensor(
                [[400.0, 0.0, 200.0], [0.0, 120.0, 100.0], [0.0, 0.0, 1.0]],
                dtype=torch.float64,
            ),
        )
    ],
)
def test_scale_intrinsics_numeric_correctness(intrinsics, resolution, expected):
    original = intrinsics.clone()
    # Pass resolution through directly (tests use resolution=(H, W))
    out = scale_intrinsics(intrinsics, resolution=resolution, scale=None, inplace=False)
    # 1) scaled values correct
    assert torch.allclose(out, expected)
    # 2) returned tensor is a new tensor
    assert out.data_ptr() != intrinsics.data_ptr()
    # 3) input intrinsics unchanged
    assert torch.allclose(intrinsics, original)


@pytest.mark.parametrize(
    "intrinsics,resolution,expected",
    [
        (
            torch.tensor(
                [[100.0, 0.0, 50.0], [0.0, 120.0, 100.0], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            ),
            (100, 200),
            torch.tensor(
                [[200.0, 0.0, 100.0], [0.0, 60.0, 50.0], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            ),
        )
    ],
)
def test_scale_intrinsics_inplace_behavior(intrinsics, resolution, expected):
    # Pass resolution through directly (tests use resolution=(H, W))
    out = scale_intrinsics(intrinsics, resolution=resolution, scale=None, inplace=True)
    assert out.data_ptr() == intrinsics.data_ptr(), "Expected in-place modification to return same tensor"
    assert torch.allclose(intrinsics, expected)
