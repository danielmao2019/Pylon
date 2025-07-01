from typing import Any, Dict
import pytest
import random
import torch
from concurrent.futures import ThreadPoolExecutor
from data.datasets.change_detection_datasets.bi_temporal.sysu_cd_dataset import SYSU_CD_Dataset


def test_sysu_cd_sha1sum() -> None:
    _ = SYSU_CD_Dataset(
        data_root="./data/datasets/soft_links/SYSU-CD", split='train',
        check_sha1sum=True,
    )


def validate_inputs(inputs: Dict[str, Any], dataset: SYSU_CD_Dataset) -> None:
    """Validate the inputs of a datapoint."""
    assert type(inputs) == dict
    assert set(inputs.keys()) == set(SYSU_CD_Dataset.INPUT_NAMES)
    img_1 = inputs['img_1']
    img_2 = inputs['img_2']
    assert type(img_1) == torch.Tensor and img_1.ndim == 3 and img_1.dtype == torch.float32
    assert type(img_2) == torch.Tensor and img_2.ndim == 3 and img_2.dtype == torch.float32
    assert img_1.shape == img_2.shape, f"{img_1.shape=}, {img_2.shape=}"


def validate_labels(labels: Dict[str, Any], class_dist: torch.Tensor, dataset: SYSU_CD_Dataset) -> None:
    """Validate the labels of a datapoint."""
    assert type(labels) == dict
    assert set(labels.keys()) == set(SYSU_CD_Dataset.LABEL_NAMES)
    change_map = labels['change_map']
    assert set(torch.unique(change_map).tolist()).issubset({0, 1}), f"{torch.unique(change_map)=}"
    for cls in range(dataset.NUM_CLASSES):
        class_dist[cls] += torch.sum(change_map == cls)


def validate_meta_info(meta_info: Dict[str, Any], datapoint_idx: int) -> None:
    """Validate the meta_info of a datapoint."""
    assert 'idx' in meta_info, f"meta_info should contain 'idx' key: {meta_info.keys()=}"
    assert meta_info['idx'] == datapoint_idx, f"meta_info['idx'] should match datapoint index: {meta_info['idx']=}, {datapoint_idx=}"


def validate_class_distribution(class_dist: torch.Tensor, dataset: SYSU_CD_Dataset, num_samples: int) -> None:
    """Validate the class distribution tensor against the dataset's expected distribution."""
    # Validate class distribution (only if we processed the full dataset)
    if num_samples == len(dataset):
        assert type(dataset.CLASS_DIST) == list, f"{type(dataset.CLASS_DIST)=}"
        assert class_dist.tolist() == dataset.CLASS_DIST, f"{class_dist=}, {dataset.CLASS_DIST=}"


@pytest.mark.parametrize('split', ['train', 'val', 'test'])
def test_sysu_cd(split: str, max_samples) -> None:
    dataset = SYSU_CD_Dataset(data_root="./data/datasets/soft_links/SYSU-CD", split=split)
    assert isinstance(dataset, torch.utils.data.Dataset)
    class_dist = torch.zeros(size=(dataset.NUM_CLASSES,), device=dataset.device)

    def validate_datapoint(idx: int) -> None:
        datapoint = dataset[idx]
        assert type(datapoint) == dict
        assert set(datapoint.keys()) == set(['inputs', 'labels', 'meta_info'])
        validate_inputs(datapoint['inputs'], dataset)
        validate_labels(datapoint['labels'], class_dist, dataset)
        validate_meta_info(datapoint['meta_info'], idx)

    num_samples = get_samples_to_test(len(dataset), max_samples, default=len(dataset))
    indices = list(range(num_samples))
    with ThreadPoolExecutor() as executor:
        executor.map(validate_datapoint, indices)

    # Validate class distribution
    validate_class_distribution(class_dist, dataset, num_samples)
