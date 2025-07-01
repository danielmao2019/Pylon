import pytest
from data.datasets.change_detection_datasets.bi_temporal.kc_3d_dataset import KC3DDataset
import torch
from tests.data.datasets.conftest import get_samples_to_test


@pytest.mark.parametrize("dataset", [
    (KC3DDataset(data_root="./data/datasets/soft_links/KC3D", split='train')),
])
def test_kc_3d(dataset: torch.utils.data.Dataset, max_samples) -> None:
    assert isinstance(dataset, torch.utils.data.Dataset)
    samples_to_test = get_samples_to_test(len(dataset), max_samples, default=100)
    for idx in range(samples_to_test):
        datapoint = dataset[idx]
        inputs, labels, meta_info = datapoint['inputs'], datapoint['labels'], datapoint['meta_info']
        assert set(inputs.keys()) == set(KC3DDataset.INPUT_NAMES), f"{set(inputs.keys())=}"
        assert set(labels.keys()) == set(KC3DDataset.LABEL_NAMES), f"{set(inputs.keys())=}"
        assert 'idx' in meta_info, f"meta_info should contain 'idx' key: {meta_info.keys()=}"
        assert meta_info['idx'] == idx, f"meta_info['idx'] should match datapoint index: {meta_info['idx']=}, {idx=}"
