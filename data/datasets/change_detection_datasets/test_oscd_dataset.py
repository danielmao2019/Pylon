import pytest
from .oscd_dataset import OSCDDataset
import os
import torch
import utils


@pytest.mark.parametrize("dataset", [
    (OSCDDataset(data_root="./data/datasets/soft_links/OSCD", split='train')),
    (OSCDDataset(data_root="./data/datasets/soft_links/OSCD", split='test')),
    (OSCDDataset(data_root="./data/datasets/soft_links/OSCD", split='train', bands=['B01', 'B02'])),
    (OSCDDataset(data_root="./data/datasets/soft_links/OSCD", split='test', bands=['B08', 'B8A'])),
])
def test_oscd(dataset: torch.utils.data.Dataset) -> None:
    assert isinstance(dataset, torch.utils.data.Dataset)
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
        assert 0 <= img_1.min() <= img_1.max() <= 1
        assert type(img_2) == torch.Tensor and img_2.ndim == 3 and img_2.dtype == torch.float32
        assert 0 <= img_2.min() <= img_2.max() <= 1
        assert img_1.shape == img_2.shape, f"{img_1.shape=}, {img_2.shape=}"
        # inspect labels
        labels = datapoint['labels']
        assert type(labels) == dict
        assert set(labels.keys()) == set(OSCDDataset.LABEL_NAMES)
        change_map = labels['change_map']
        assert set(torch.unique(change_map).tolist()) == set([0, 1]), f"{torch.unique(change_map)=}"
        # sanity check for consistency between different modalities
        for input_idx in [1, 2]:
            tif_input = utils.io.load_image(filepaths=list(filter(
                lambda x: os.path.splitext(os.path.basename(x))[0].split('_')[-1] in ['B04', 'B03', 'B02'],
                dataset.annotations[idx]['inputs'][f'tif_input_{input_idx}_filepaths'],
            )), dtype=torch.float32, sub=None, div=None)
            png_input = utils.io.load_image(
                filepath=dataset.annotations[idx]['inputs'][f'png_input_{input_idx}_filepath'],
                dtype=torch.float32, sub=None, div=255.0,
            )
            assert torch.equal(tif_input, png_input)
        tif_label = utils.io.load_image(
            filepaths=[dataset.annotations[idx]['labels']['tif_label_filepaths']],
            dtype=torch.int64, sub=1, div=None,
        )
        png_label = (torch.mean(utils.io.load_image(
            filepath=dataset.annotations[idx]['png_label_filepath'],
            dtype=torch.int64, sub=None, div=None,
        )[:3, 0, 0], dim=0, keepdim=True) > 0.5).to(torch.int64)
        assert torch.equal(tif_label, png_label)
