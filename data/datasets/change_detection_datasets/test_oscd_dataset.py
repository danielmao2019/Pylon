import pytest
from .oscd_dataset import OSCDDataset
import torch


@pytest.mark.parametrize("dataset", [
    (OSCDDataset(data_root="./data/datasets/soft_links/OSCD", split='train')),
    (OSCDDataset(data_root="./data/datasets/soft_links/OSCD", split='test')),
])
def test_oscd(dataset: torch.utils.data.Dataset) -> None:
    assert isinstance(dataset, torch.utils.data.Dataset)
    for i in range(len(dataset)):
        _ = dataset[i]
