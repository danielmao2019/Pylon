import pytest
from models.change_detection.fc_siam.fully_convolutional_siamese_networks import FullyConvolutionalSiameseNetwork
import torch


@pytest.mark.parametrize("arch", [
    ('FC-EF'),
    ('FC-Siam-conc'),
    ('FC-Siam-diff'),
])
def test_fully_convolutional_siamese_networks(arch: str) -> None:
    model = FullyConvolutionalSiameseNetwork(
        arch=arch, num_classes=2,
        in_channels=3 if arch != 'FC-EF' else 6,
    )
    inputs = {
        'img_1': torch.zeros(size=(1, 3, 32, 32)),
        'img_2': torch.zeros(size=(1, 3, 32, 32)),
    }
    model(inputs)
