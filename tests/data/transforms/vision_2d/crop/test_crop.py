import pytest
from data.transforms.vision_2d.crop.crop import Crop
import torch


test_tensor = torch.Tensor([
    [7.0826e-01, 5.8945e-01, 3.6625e-01, 6.1209e-01],
    [7.0273e-02, 8.1569e-04, 4.3327e-01, 2.3406e-01],
    [6.8771e-01, 4.1768e-01, 8.8121e-01, 3.5236e-01],
    [7.9153e-01, 8.4036e-01, 3.4045e-01, 3.6258e-02],
])


@pytest.mark.parametrize("loc, size, expected", [
    (
        (1, 1), (2, 2),
        torch.Tensor([[8.1569e-04, 4.3327e-01], [4.1768e-01, 8.8121e-01]])
    ),
    (
        (0, 0), (2, 2),
        torch.Tensor([[7.0826e-01, 5.8945e-01], [7.0273e-02, 8.1569e-04]])
    ),
    (
        (2, 1), (2, 2),
        torch.Tensor([[4.3327e-01, 2.3406e-01], [8.8121e-01, 3.5236e-01]])
    ),
])
def test_crop(loc, size, expected) -> None:
    # Initialize Crop and perform cropping
    crop_op = Crop(loc, size)
    cropped_tensor = crop_op(test_tensor)

    # Validate the result
    assert torch.allclose(cropped_tensor, expected), f"Expected {expected}, but got {cropped_tensor}"


@pytest.mark.parametrize("loc, size, expected_exception", [
    ((-1, 0), (2, 2), ValueError),  # Invalid loc
    ((1, 1), (-2, 2), ValueError),  # Invalid size
    ((3, 3), (2, 2), AssertionError),  # Out-of-bounds crop
])
def test_invalid_crop(loc, size, expected_exception) -> None:
    with pytest.raises(expected_exception):
        crop_op = Crop(loc, size)
        crop_op(test_tensor)  # This line will only be reached for out-of-bounds cases
