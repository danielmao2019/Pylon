import pytest
from data.datasets.semantic_segmentation_datasets.whu_bd_dataset import WHU_BD_Dataset
import torch


def test_whu_bd_sha1sum() -> None:
    _ = WHU_BD_Dataset(
        data_root="./data/datasets/soft_links/WHU-BD", split='train',
        check_sha1sum=True,
    )


@pytest.mark.parametrize("dataset", [
    (WHU_BD_Dataset(data_root="./data/datasets/soft_links/WHU-BD", split='train')),
    (WHU_BD_Dataset(data_root="./data/datasets/soft_links/WHU-BD", split='val')),
    (WHU_BD_Dataset(data_root="./data/datasets/soft_links/WHU-BD", split='test')),
])
def test_whu_bd_dataset(dataset) -> None:
    assert isinstance(dataset, torch.utils.data.Dataset)
    for idx in range(len(dataset)):
        datapoint = dataset[idx]
        assert type(datapoint) == dict
        assert set(datapoint.keys()) == set(['inputs', 'labels', 'meta_info'])
        # inspect inputs
        inputs = datapoint['inputs']
        assert type(inputs) == dict
        assert set(inputs.keys()) == set(WHU_BD_Dataset.INPUT_NAMES)
        image = inputs['image']
        assert type(image) == torch.Tensor, f"{type(image)=}"
        assert image.ndim == 3 and image.size(0) == 3, f"{image.shape=}"
        assert image.dtype == torch.float32, f"{image.dtype=}"
        # inspect labels
        labels = datapoint['labels']
        assert type(labels) == dict
        assert set(labels.keys()) == set(WHU_BD_Dataset.LABEL_NAMES)
        semantic_map = labels['semantic_map']
        assert type(semantic_map) == torch.Tensor, f"{type(semantic_map)=}"
        assert semantic_map.ndim == 2, f"{semantic_map.shape=}"
        assert semantic_map.dtype == torch.int64, f"{semantic_map.dtype=}"
        assert set(torch.unique(semantic_map).tolist()).issubset({0, 1}), f"{torch.unique(semantic_map)=}"
