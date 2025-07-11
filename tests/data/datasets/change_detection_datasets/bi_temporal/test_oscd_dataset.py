from typing import Any, Dict
import pytest
import torch
from concurrent.futures import ThreadPoolExecutor
from data.datasets.change_detection_datasets.bi_temporal.oscd_dataset import OSCDDataset
import utils


def validate_inputs(inputs: Dict[str, Any], dataset: OSCDDataset) -> None:
    """Validate the inputs of a datapoint."""
    assert type(inputs) == dict
    assert set(inputs.keys()) == set(OSCDDataset.INPUT_NAMES)
    img_1 = inputs['img_1']
    img_2 = inputs['img_2']
    assert type(img_1) == torch.Tensor and img_1.ndim == 3 and img_1.dtype == torch.float32
    assert type(img_2) == torch.Tensor and img_2.ndim == 3 and img_2.dtype == torch.float32
    if dataset.bands is None:
        for input_idx, x in enumerate([img_1, img_2]):
            assert 0 <= x.min() <= x.max() <= 1, f"{input_idx=}, {x.min()=}, {x.max()=}"


def validate_labels(labels: Dict[str, Any], class_dist: torch.Tensor, dataset: OSCDDataset, idx: int) -> None:
    """Validate the labels of a datapoint."""
    assert type(labels) == dict
    assert set(labels.keys()) == set(OSCDDataset.LABEL_NAMES)
    change_map = labels['change_map']
    assert set(torch.unique(change_map).tolist()) == set([0, 1]), f"{torch.unique(change_map)=}"
    for cls in range(dataset.NUM_CLASSES):
        class_dist[cls] += torch.sum(change_map == cls)

    # sanity check for consistency between different modalities
    # TODO: Enable the following assertions
    # for input_idx in [1, 2]:
    #     tif_input = utils.io.load_image(filepaths=list(filter(
    #         lambda x: os.path.splitext(os.path.basename(x))[0].split('_')[-1] in ['B04', 'B03', 'B02'],
    #         dataset.annotations[idx]['inputs'][f'tif_input_{input_idx}_filepaths'],
    #     ))[::-1], dtype=torch.float32, sub=None, div=None)
    #     lower = torch.quantile(tif_input, 0.02)
    #     upper = torch.quantile(tif_input, 0.98)
    #     tif_input = (tif_input - lower) / (upper - lower)
    #     tif_input = torch.clamp(tif_input, min=0, max=1)
    #     tif_input = (tif_input * 255).to(torch.uint8)
    #     png_input = utils.io.load_image(
    #         filepath=dataset.annotations[idx]['inputs'][f'png_input_{input_idx}_filepath'],
    #         dtype=torch.uint8, sub=None, div=None,
    #     )
    #     assert torch.equal(tif_input, png_input)
    tif_label = utils.io.load_image(
        filepaths=dataset.annotations[idx]['labels']['tif_label_filepaths'],
        dtype=torch.int64, sub=1, div=None,
    )
    png_label = utils.io.load_image(
        filepath=dataset.annotations[idx]['labels']['png_label_filepath'],
        dtype=torch.float32, sub=None, div=None,
    )
    if png_label.ndim == 3:
        assert png_label.shape[0] in {3, 4}, f"{png_label.shape=}"
        png_label = torch.mean(png_label[:3, :, :], dim=0, keepdim=True)
    png_label = (png_label > 0.5).to(torch.int64)
    assert torch.sum(tif_label != png_label) / torch.numel(tif_label) < 0.003, \
        f"{torch.sum(tif_label != png_label) / torch.numel(tif_label)=}"


def validate_meta_info(meta_info: Dict[str, Any], datapoint_idx: int) -> None:
    """Validate the meta_info of a datapoint."""
    assert 'idx' in meta_info, f"meta_info should contain 'idx' key: {meta_info.keys()=}"
    assert meta_info['idx'] == datapoint_idx, f"meta_info['idx'] should match datapoint index: {meta_info['idx']=}, {datapoint_idx=}"


def validate_class_distribution(class_dist: torch.Tensor, dataset: OSCDDataset, num_samples: int) -> None:
    """Validate the class distribution tensor against the dataset's expected distribution."""
    # Validate class distribution (only if we processed the full dataset)
    if num_samples == len(dataset):
        assert type(dataset.CLASS_DIST) == list, f"{type(dataset.CLASS_DIST)=}"
        assert class_dist.tolist() == dataset.CLASS_DIST, f"{class_dist=}, {dataset.CLASS_DIST=}"


@pytest.fixture
def dataset(request):
    """Fixture for creating an OSCDDataset instance."""
    split, bands = request.param
    return OSCDDataset(data_root="./data/datasets/soft_links/OSCD", split=split, bands=bands)


@pytest.mark.parametrize('dataset', [
    ('train', None),
    ('test', None),
    ('train', ['B01', 'B02']),
    ('test', ['B01', 'B02']),
    ('train', ['B04', 'B03', 'B02']),
    ('train', ['B08', 'B8A']),
], indirect=True)
def test_oscd(dataset, max_samples, get_samples_to_test) -> None:
    assert isinstance(dataset, torch.utils.data.Dataset)
    assert len(dataset) > 0, "Dataset should not be empty"
    class_dist = torch.zeros(size=(dataset.NUM_CLASSES,), device=dataset.device)

    def validate_datapoint(idx: int) -> None:
        datapoint = dataset[idx]
        assert type(datapoint) == dict
        assert set(datapoint.keys()) == set(['inputs', 'labels', 'meta_info'])
        validate_inputs(datapoint['inputs'], dataset)
        validate_labels(datapoint['labels'], class_dist, dataset, idx)
        validate_meta_info(datapoint['meta_info'], idx)

    num_samples = get_samples_to_test(len(dataset), max_samples, default=len(dataset))
    indices = list(range(num_samples))
    with ThreadPoolExecutor() as executor:
        executor.map(validate_datapoint, indices)

    # Validate class distribution
    validate_class_distribution(class_dist, dataset, num_samples)
