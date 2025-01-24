import pytest
from .oscd_dataset import OSCDDataset
import os
import torch
import utils


@pytest.mark.parametrize("dataset", [
    (OSCDDataset(data_root="./data/datasets/soft_links/OSCD", split='train')),
    (OSCDDataset(data_root="./data/datasets/soft_links/OSCD", split='test')),
    (OSCDDataset(data_root="./data/datasets/soft_links/OSCD", split='train', bands=['B01', 'B02'])),
    (OSCDDataset(data_root="./data/datasets/soft_links/OSCD", split='test', bands=['B01', 'B02'])),
    (OSCDDataset(data_root="./data/datasets/soft_links/OSCD", split='train', bands=['B04', 'B03', 'B02'])),
    (OSCDDataset(data_root="./data/datasets/soft_links/OSCD", split='train', bands=['B08', 'B8A'])),
])
def test_oscd(dataset: torch.utils.data.Dataset) -> None:
    assert isinstance(dataset, torch.utils.data.Dataset)
    class_dist = torch.zeros(size=(dataset.NUM_CLASSES,), device=dataset.device)
    for idx in range(len(dataset)):
        datapoint = dataset[idx]
        assert type(datapoint) == dict
        assert set(datapoint.keys()) == set(['inputs', 'labels', 'meta_info'])
        # inspect inputs
        inputs = datapoint['inputs']
        assert type(inputs) == dict
        assert set(inputs.keys()) == set(OSCDDataset.INPUT_NAMES)
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
    assert class_dist.tolist() == dataset.CLASS_DIST, f"{class_dist=}, {dataset.CLASS_DIST=}"
