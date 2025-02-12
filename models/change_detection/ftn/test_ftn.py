import torch
import models


def test_ftn() -> None:
    inputs = {
        f'img_{idx}': torch.zeros(size=(1, 3, 224, 224))
        for idx in [1, 2]
    }
    model = models.change_detection.FTN()
    _ = model(inputs)
