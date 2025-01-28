import pytest
from .oscd_dataset import SYSU_CD_Dataset
import torch


@pytest.mark.parametrize("dataset", [
    (SYSU_CD_Dataset(data_root="./data/datasets/soft_links/SYSU-CD", split='train')),
    (SYSU_CD_Dataset(data_root="./data/datasets/soft_links/SYSU-CD", split='val')),
    (SYSU_CD_Dataset(data_root="./data/datasets/soft_links/SYSU-CD", split='test')),
])
def test_sysu_cd(dataset: torch.utils.data.Dataset) -> None:
    assert isinstance(dataset, torch.utils.data.Dataset)
    class_dist = torch.zeros(size=(dataset.NUM_CLASSES,), device=dataset.device)
    for idx in range(len(dataset)):
        datapoint = dataset[idx]
        assert type(datapoint) == dict
        assert set(datapoint.keys()) == set(['inputs', 'labels', 'meta_info'])
        # inspect inputs
        inputs = datapoint['inputs']
        assert type(inputs) == dict
        assert set(inputs.keys()) == set(SYSU_CD_Dataset.INPUT_NAMES)
        img_1 = inputs['img_1']
        img_2 = inputs['img_2']
        assert type(img_1) == torch.Tensor and img_1.ndim == 3 and img_1.dtype == torch.float32
        assert type(img_2) == torch.Tensor and img_2.ndim == 3 and img_2.dtype == torch.float32
        assert img_1.shape == img_2.shape, f"{img_1.shape=}, {img_2.shape=}"
        if dataset.bands is None:
            for input_idx, x in enumerate([img_1, img_2]):
                assert 0 <= x.min() <= x.max() <= 1, f"{input_idx=}, {x.min()=}, {x.max()=}"
        # inspect labels
        labels = datapoint['labels']
        assert type(labels) == dict
        assert set(labels.keys()) == set(SYSU_CD_Dataset.LABEL_NAMES)
        change_map = labels['change_map']
        assert set(torch.unique(change_map).tolist()).issubset({0, 1}), f"{torch.unique(change_map)=}"
        for cls in range(dataset.NUM_CLASSES):
            class_dist[cls] += torch.sum(change_map == cls)
    assert type(dataset.CLASS_DIST) == list, f"{type(dataset.CLASS_DIST)=}"
    assert class_dist.tolist() == dataset.CLASS_DIST, f"{class_dist=}, {dataset.CLASS_DIST=}"
