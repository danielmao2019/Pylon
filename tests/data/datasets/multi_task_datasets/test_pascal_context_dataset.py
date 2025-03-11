import pytest
from data.datasets.multi_task_datasets.pascal_context_dataset import PASCALContextDataset
import torch


@pytest.mark.parametrize("dataset", [
    (PASCALContextDataset(data_root="./data/datasets/soft_links/PASCAL_MT", split='train')),
    (PASCALContextDataset(data_root="./data/datasets/soft_links/PASCAL_MT", split='val')),
])
def test_pascal_context_dataset(dataset: torch.utils.data.Dataset) -> None:
    assert isinstance(dataset, torch.utils.data.Dataset)
    for i in range(min(len(dataset), 3)):
        datapoint = dataset[i]
