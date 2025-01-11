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
        datapoint = dataset[i]
        assert type(datapoint) == dict
        assert set(datapoint.keys()) == set(['inputs', 'labels', 'meta_info'])
        # inspect inputs
        inputs = datapoint['inputs']
        assert type(inputs) == dict
        assert set(inputs.keys()) == set(OSCDDataset.INPUT_NAMES)
        img_1 = inputs['img_1']
        img_2 = inputs['img_2']
        assert type(img_1) == torch.Tensor and img_1.ndim == 3 and img_1.dtype == torch.float32
        assert 0 <= img_1.min() <= img_1.max() <= 1
        assert type(img_2) == torch.Tensor and img_2.ndim == 3 and img_2.dtype == torch.float32
        assert 0 <= img_2.min() <= img_2.max() <= 1
        assert img_1.shape == img_2.shape, f"{img_1.shape=}, {img_2.shape=}"
        # inspect labels
        labels = datapoint['labels']
        assert type(labels) == dict
        assert set(labels.keys()) == set(OSCDDataset.LABEL_NAMES)
        change_map = labels['change_map']
        assert set(torch.unique(change_map).tolist()) == set([0, 1]), f"{torch.unique(change_map)=}"
