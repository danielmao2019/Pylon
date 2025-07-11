from typing import Dict, Any
import pytest
import torch
from concurrent.futures import ThreadPoolExecutor
from data.datasets.change_detection_datasets.single_temporal.i3pe_dataset import I3PEDataset
from data.datasets import Bi2SingleTemporal, SYSU_CD_Dataset


def validate_inputs(inputs: Dict[str, Any]) -> None:
    assert isinstance(inputs, dict), f"{type(inputs)=}"
    assert set(inputs.keys()) == set(I3PEDataset.INPUT_NAMES)
    img_1 = inputs['img_1']
    img_2 = inputs['img_2']
    assert type(img_1) == torch.Tensor and img_1.ndim == 3 and img_1.dtype == torch.float32
    assert type(img_2) == torch.Tensor and img_2.ndim == 3 and img_2.dtype == torch.float32
    assert img_1.shape == img_2.shape, f"{img_1.shape=}, {img_2.shape=}"


def validate_labels(labels: Dict[str, Any]) -> None:
    assert isinstance(labels, dict), f"{type(labels)=}"
    assert set(labels.keys()) == set(I3PEDataset.LABEL_NAMES)
    change_map = labels['change_map']
    assert type(change_map) == torch.Tensor and change_map.ndim == 2 and change_map.dtype == torch.int64
    assert set(torch.unique(change_map).tolist()).issubset({0, 1}), f"{torch.unique(change_map)=}"


def validate_meta_info(meta_info: Dict[str, Any], datapoint_idx: int) -> None:
    assert isinstance(meta_info, dict), f"{type(meta_info)=}"
    assert 'idx' in meta_info, f"meta_info should contain 'idx' key: {meta_info.keys()=}"
    assert meta_info['idx'] == datapoint_idx, f"meta_info['idx'] should match datapoint index: {meta_info['idx']=}, {datapoint_idx=}"


@pytest.fixture
def dataset(request):
    """Fixture for creating an I3PEDataset instance."""
    source = Bi2SingleTemporal(SYSU_CD_Dataset(data_root="./data/datasets/soft_links/SYSU-CD", split='train'))
    return I3PEDataset(source=source, dataset_size=len(source), exchange_ratio=0.75)


def test_i3pe_dataset(dataset, max_samples, get_samples_to_test) -> None:
    assert isinstance(dataset, torch.utils.data.Dataset), f"Expected torch.utils.data.Dataset, got {type(dataset)}"
    assert len(dataset) > 0, "Dataset should not be empty"

    def validate_datapoint(idx: int) -> None:
        datapoint = dataset[idx]
        assert isinstance(datapoint, dict), f"{type(datapoint)=}"
        assert datapoint.keys() == {'inputs', 'labels', 'meta_info'}
        validate_inputs(datapoint['inputs'])
        validate_labels(datapoint['labels'])
        validate_meta_info(datapoint['meta_info'], idx)

    num_samples = get_samples_to_test(len(dataset), max_samples, default=100)
    indices = list(range(num_samples))

    with ThreadPoolExecutor() as executor:
        executor.map(validate_datapoint, indices)
