import pytest
from .xview2_dataset import xView2Dataset
import torch


@pytest.mark.parametrize("dataset", [
    (xView2Dataset(data_root="./data/datasets/soft_links/xView2", split='train')),
    (xView2Dataset(data_root="./data/datasets/soft_links/xView2", split='test')),
    (xView2Dataset(data_root="./data/datasets/soft_links/xView2", split='hold')),
])
def test_xview2(dataset: torch.utils.data.Dataset) -> None:
    assert isinstance(dataset, torch.utils.data.Dataset)
    for idx in range(len(dataset)):
        datapoint = dataset[idx]
        # inspect inputs
        inputs = datapoint['inputs']
        assert set(inputs.keys()) == set(xView2Dataset.INPUT_NAMES)
        assert 0 <= inputs['img_1'].min() <= inputs['img_1'].max() <= 1
        assert 0 <= inputs['img_2'].min() <= inputs['img_2'].max() <= 1
        # inspect labels
        labels = datapoint['labels']
        assert set(labels.keys()) == set(xView2Dataset.LABEL_NAMES)
        assert set(torch.unique(labels['lbl_1']).tolist()).issubset(set([0, 1, 2, 3, 4])), f"{torch.unique(labels['lbl_1'])=}"
        assert set(torch.unique(labels['lbl_2']).tolist()).issubset(set([0, 1, 2, 3, 4])), f"{torch.unique(labels['lbl_2'])=}"
