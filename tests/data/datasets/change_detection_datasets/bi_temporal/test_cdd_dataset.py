from typing import Any, Dict
import pytest
import torch
from concurrent.futures import ThreadPoolExecutor
from data.datasets.change_detection_datasets.bi_temporal.cdd_dataset import CDDDataset


def validate_inputs(inputs: Dict[str, Any], dataset: CDDDataset, idx: int) -> None:
    """Validate the inputs of a datapoint."""
    assert isinstance(inputs, dict), f"Inputs at index {idx} are not a dictionary."
    assert set(inputs.keys()) == set(CDDDataset.INPUT_NAMES), \
        f"Unexpected input keys at index {idx}: {inputs.keys()}"

    img_1 = inputs['img_1']
    img_2 = inputs['img_2']
    assert isinstance(img_1, torch.Tensor), f"img_1 at index {idx} is not a Tensor."
    assert isinstance(img_2, torch.Tensor), f"img_2 at index {idx} is not a Tensor."
    assert img_1.ndim == 3 and img_2.ndim == 3, f"Input images at index {idx} must be 3D tensors."
    assert img_1.dtype == torch.float32 and img_2.dtype == torch.float32, \
        f"Input images at index {idx} must have dtype torch.float32."
    assert img_1.shape == img_2.shape, \
        f"Shape mismatch between img_1 and img_2 at index {idx}: {img_1.shape} vs {img_2.shape}"


def validate_labels(labels: Dict[str, Any], class_dist: torch.Tensor, dataset: CDDDataset, idx: int) -> None:
    """Validate the labels of a datapoint."""
    assert isinstance(labels, dict), f"Labels at index {idx} are not a dictionary."
    assert set(labels.keys()) == set(CDDDataset.LABEL_NAMES), \
        f"Unexpected label keys at index {idx}: {labels.keys()}"

    change_map = labels['change_map']
    assert isinstance(change_map, torch.Tensor), f"Change map at index {idx} is not a Tensor."
    assert change_map.ndim == 2, f"Change map at index {idx} must be a 2D tensor."
    assert change_map.dtype == torch.int64, f"Change map at index {idx} must have dtype torch.int64."
    unique_values = set(torch.unique(change_map).tolist())
    assert unique_values.issubset({0, 1}), \
        f"Unexpected values in change map at index {idx}: {unique_values}"
    for cls in range(dataset.NUM_CLASSES):
        class_dist[cls] += torch.sum(change_map == cls)


def validate_meta_info(meta_info: Dict[str, Any], datapoint_idx: int) -> None:
    """Validate the meta_info of a datapoint."""
    assert isinstance(meta_info, dict), f"Meta info at index {datapoint_idx} is not a dictionary."
    assert 'idx' in meta_info, f"meta_info should contain 'idx' key: {meta_info.keys()=}"
    assert meta_info['idx'] == datapoint_idx, f"meta_info['idx'] should match datapoint index: {meta_info['idx']=}, {datapoint_idx=}"
    assert 'image_resolution' in meta_info, f"Missing 'image_resolution' in meta info at index {datapoint_idx}."


def validate_class_distribution(class_dist: torch.Tensor, dataset: CDDDataset, num_samples: int) -> None:
    """Validate the class distribution tensor against the dataset's expected distribution."""
    # Validate class distribution (only if we processed the full dataset)
    if num_samples == len(dataset):
        assert type(dataset.CLASS_DIST) == list, f"{type(dataset.CLASS_DIST)=}"
        assert class_dist.tolist() == dataset.CLASS_DIST, f"{class_dist=}, {dataset.CLASS_DIST=}"


@pytest.fixture
def dataset(request):
    """Fixture for creating a CDDDataset instance."""
    split = request.param
    return CDDDataset(data_root="./data/datasets/soft_links/CDD", split=split)


@pytest.mark.parametrize('dataset', ['train', 'val', 'test'], indirect=True)
def test_cdd_dataset(dataset, max_samples, get_samples_to_test) -> None:
    assert isinstance(dataset, torch.utils.data.Dataset), "Dataset is not a valid PyTorch dataset instance."
    assert len(dataset) > 0, "Dataset should not be empty"
    class_dist = torch.zeros(size=(dataset.NUM_CLASSES,), dtype=torch.int64, device=dataset.device)

    def validate_datapoint(idx: int) -> None:
        datapoint = dataset[idx]
        assert isinstance(datapoint, dict), f"Datapoint at index {idx} is not a dictionary."
        assert set(datapoint.keys()) == {'inputs', 'labels', 'meta_info'}, \
            f"Unexpected keys in datapoint at index {idx}: {datapoint.keys()}"
        validate_inputs(datapoint['inputs'], dataset, idx)
        validate_labels(datapoint['labels'], class_dist, dataset, idx)
        validate_meta_info(datapoint['meta_info'], idx)

    num_samples = get_samples_to_test(len(dataset), max_samples, default=len(dataset))
    indices = list(range(num_samples))
    with ThreadPoolExecutor() as executor:
        executor.map(validate_datapoint, indices)

    # Validate class distribution
    validate_class_distribution(class_dist, dataset, num_samples)
