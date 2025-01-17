import pytest
from .levir_cd_dataset import LevirCdDataset
import os
import torch
import utils

@pytest.mark.parametrize("dataset", [
    (LevirCdDataset(data_root="./data/datasets/soft_links/LEVIR_CD", split='train')),
    (LevirCdDataset(data_root="./data/datasets/soft_links/LEVIR_CD", split='test')),
    (LevirCdDataset(data_root="./data/datasets/soft_links/LEVIR_CD", split='val')),
])
def test_levir_cd(dataset: torch.utils.data.Dataset) -> None:
    assert isinstance(dataset, torch.utils.data.Dataset)
    for idx in range(len(dataset)):
        datapoint = dataset[idx]
        assert type(datapoint) == dict
        assert set(datapoint.keys()) == set(['inputs', 'labels', 'meta_info'])
        # inspect inputs
        inputs = datapoint['inputs']
        assert type(inputs) == dict
        assert set(inputs.keys()) == set(LevirCdDataset.INPUT_NAMES)
        img_1 = inputs['img_1']
        img_2 = inputs['img_2']
        assert type(img_1) == torch.Tensor and img_1.ndim == 3 and img_1.dtype == torch.float32
        assert type(img_2) == torch.Tensor and img_2.ndim == 3 and img_2.dtype == torch.float32
        assert img_1.shape == img_2.shape, f"{img_1.shape=}, {img_2.shape=}"
        # inspect labels
        labels = datapoint['labels']
        assert type(labels) == dict
        assert set(labels.keys()) == set(LevirCdDataset.LABEL_NAMES)
        change_map = labels['change_map']
        assert set(torch.unique(change_map).tolist()) == set([0, 1]), f"{torch.unique(change_map)=}"