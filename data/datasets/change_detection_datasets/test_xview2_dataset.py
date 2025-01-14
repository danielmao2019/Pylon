import pytest
from .xview2_dataset import xView2Dataset
import torch


@pytest.mark.parametrize("dataset", [
    (xView2Dataset(data_root="./data/datasets/soft_links/xView2", split='train')),
    (xView2Dataset(data_root="./data/datasets/soft_links/xView2", split='test')),
    (xView2Dataset(data_root="./data/datasets/soft_links/xView2", split='hold')),
    (xView2Dataset(data_root="./data/datasets/soft_links/xView2", split='train', pre_or_post_disaster='pre')),
])
def test_xview2(dataset: torch.utils.data.Dataset) -> None:
    assert isinstance(dataset, torch.utils.data.Dataset)
    for idx in range(len(dataset)):
        datapoint = dataset[idx]
