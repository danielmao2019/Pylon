import pytest
from .kc_3d_dataset import KC3DDataset
import torch


@pytest.mark.parametrize("dataset", [
    (KC3DDataset(data_root="./data/datasets/soft_links/KC3D", split='train')),
])
def test_kc_3d(dataset: torch.utils.data.Dataset) -> None:
    assert isinstance(dataset, torch.utils.data.Dataset)
    for idx in range(min(len(dataset), 100)):
        datapoint = dataset[idx]
        inputs, labels, meta_info = datapoint['inputs'], datapoint['labels'], datapoint['meta_info']
        assert set(inputs.keys()) == set(KC3DDataset.INPUT_NAMES), f"{set(inputs.keys())=}"
        assert set(labels.keys()) == set(KC3DDataset.LABEL_NAMES), f"{set(inputs.keys())=}"
        assert meta_info == {}
