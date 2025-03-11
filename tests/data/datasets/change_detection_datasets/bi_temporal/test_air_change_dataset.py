import pytest
from data.datasets.change_detection_datasets.bi_temporal.air_change_dataset import AirChangeDataset
import torch


@pytest.mark.parametrize("dataset", [
    AirChangeDataset(data_root="./data/datasets/soft_links/AirChange", split='train'),
    AirChangeDataset(data_root="./data/datasets/soft_links/AirChange", split='test'),
])
def test_air_change(dataset: torch.utils.data.Dataset) -> None:
    assert isinstance(dataset, torch.utils.data.Dataset), "Dataset must inherit from torch.utils.data.Dataset"

    # Initialize class distribution tensor
    class_dist = torch.zeros(size=(dataset.NUM_CLASSES,), device=dataset.device)

    # Iterate through dataset
    for idx in range(len(dataset)):
        datapoint = dataset[idx]

        # Validate datapoint structure
        assert isinstance(datapoint, dict), f"Expected datapoint to be a dict, got {type(datapoint)}"
        assert set(datapoint.keys()) == {"inputs", "labels", "meta_info"}, f"Unexpected keys in datapoint: {datapoint.keys()}"

        # Validate inputs
        _validate_inputs(datapoint['inputs'], dataset)

        # Validate labels
        _validate_labels(datapoint['labels'], class_dist, dataset)

    # Validate class distribution
    _validate_class_distribution(class_dist, dataset)


def _validate_inputs(inputs: dict, dataset: AirChangeDataset) -> None:
    """Validate the inputs of a datapoint."""
    assert isinstance(inputs, dict), f"Expected inputs to be a dict, got {type(inputs)}"
    assert set(inputs.keys()) == set(dataset.INPUT_NAMES), f"Unexpected input keys: {inputs.keys()}"

    for img_name in ["img_1", "img_2"]:
        img = inputs[img_name]
        assert isinstance(img, torch.Tensor), f"{img_name} should be a torch.Tensor, got {type(img)}"
        assert img.ndim == 3, f"{img_name} should have 3 dimensions, got {img.ndim}"
        assert img.size(0) == 3, f"{img_name} should have 3 channels (C x H x W), got {img.size(0)}"
        assert img.dtype == torch.float32, f"{img_name} should be of dtype torch.float32, got {img.dtype}"


def _validate_labels(labels: dict, class_dist: torch.Tensor, dataset: AirChangeDataset) -> None:
    """Validate the labels of a datapoint."""
    assert isinstance(labels, dict), f"Expected labels to be a dict, got {type(labels)}"
    assert set(labels.keys()) == set(dataset.LABEL_NAMES), f"Unexpected label keys: {labels.keys()}"

    change_map = labels['change_map']
    assert isinstance(change_map, torch.Tensor), f"Expected change_map to be a torch.Tensor, got {type(change_map)}"
    unique_values = torch.unique(change_map).tolist()
    assert set(unique_values).issubset({0, 1}), f"Unexpected values in change_map: {unique_values}"

    # Update class distribution
    for cls in range(dataset.NUM_CLASSES):
        class_dist[cls] += torch.sum(change_map == cls)


def _validate_class_distribution(class_dist: torch.Tensor, dataset: AirChangeDataset) -> None:
    """Validate the class distribution tensor against the dataset's expected distribution."""
    assert isinstance(dataset.CLASS_DIST, list), f"CLASS_DIST should be a list, got {type(dataset.CLASS_DIST)}"
    if dataset.split == 'train':
        assert abs(class_dist[1] / class_dist[0] - dataset.CLASS_DIST[1] / dataset.CLASS_DIST[0]) < 1.0e-02
    else:
        assert class_dist.tolist() == dataset.CLASS_DIST, f"Class distribution mismatch: {class_dist=}, {dataset.CLASS_DIST=}"
