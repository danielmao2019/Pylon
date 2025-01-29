from .i3pe_dataset import I3PEDataset
import torch
from data.datasets import Bi2SingleTemporal, SYSU_CD_Dataset


def test_i3pe_dataset() -> None:
    source = Bi2SingleTemporal(SYSU_CD_Dataset(data_root="./data/datasets/soft_links/SYSU-CD", split='train'))
    dataset = I3PEDataset(source=source, dataset_size=len(source), exchange_ratio=0.1)
    for idx in range(min(len(dataset), 100)):
        datapoint = dataset[idx]
        assert type(datapoint) == dict
        assert set(datapoint.keys()) == set(['inputs', 'labels', 'meta_info'])
        # inspect inputs
        inputs = datapoint['inputs']
        assert type(inputs) == dict
        assert set(inputs.keys()) == set(I3PEDataset.INPUT_NAMES)
        img_1 = inputs['img_1']
        img_2 = inputs['img_2']
        assert type(img_1) == torch.Tensor and img_1.ndim == 3 and img_1.dtype == torch.float32
        assert type(img_2) == torch.Tensor and img_2.ndim == 3 and img_2.dtype == torch.float32
        assert img_1.shape == img_2.shape, f"{img_1.shape=}, {img_2.shape=}"
        # inspect labels
        labels = datapoint['labels']
        assert type(labels) == dict
        assert set(labels.keys()) == set(I3PEDataset.LABEL_NAMES)
        change_map = labels['change_map']
        assert type(change_map) == torch.Tensor and change_map.ndim == 2 and change_map.dtype == torch.int64
        assert set(torch.unique(change_map).tolist()).issubset({0, 1}), f"{torch.unique(change_map)=}"
