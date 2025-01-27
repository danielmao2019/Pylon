import pytest
from .crop import Crop
import torch


def test_crop() -> None:
    a = torch.Tensor([
        [7.0826e-01, 5.8945e-01, 3.6625e-01, 6.1209e-01],
        [7.0273e-02, 8.1569e-04, 4.3327e-01, 2.3406e-01],
        [6.8771e-01, 4.1768e-01, 8.8121e-01, 3.5236e-01],
        [7.9153e-01, 8.4036e-01, 3.4045e-01, 3.6258e-02],
    ])
    crop_op = Crop()
