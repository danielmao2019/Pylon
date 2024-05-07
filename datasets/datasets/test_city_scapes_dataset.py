import pytest
from .city_scapes_dataset import CityScapesDataset
import os
import torch


@pytest.mark.parametrize("dataset", [
    (CityScapesDataset(data_root="./datasets/datasets/soft_links/city-scapes", split='train')),
    (CityScapesDataset(data_root="./datasets/datasets/soft_links/city-scapes", split='train', indices=[0, 2, 4, 6, 8])),
])
def test_city_scapes(dataset: torch.utils.data.Dataset) -> None:
    assert isinstance(dataset, torch.utils.data.Dataset)
    for i in range(min(len(dataset), 3)):
        example = dataset[i]
        assert type(example) == dict
        assert set(example.keys()) == set(['inputs', 'labels', 'meta_info'])
        # inspect inputs
        inputs = example['inputs']
        assert type(inputs) == dict
        assert set(inputs.keys()) == set(['image'])
        image = inputs['image']
        assert type(image) == torch.Tensor
        assert len(image.shape) == 3 and image.shape[0] == 3, f"{image.shape=}"
        assert image.dtype == torch.float32
        assert -1 <= image.min() <= image.max() <= +1, f"{image.min()=}, {image.max()=}"
        # inspect labels
        labels = example['labels']
        assert type(labels) == dict
        assert set(labels.keys()) == set(['depth_estimation', 'semantic_segmentation', 'instance_segmentation'])
        depth_estimation = labels['depth_estimation']
        assert type(depth_estimation) == torch.Tensor
        assert len(depth_estimation.shape) == 2
        assert depth_estimation.dtype == torch.float32
        semantic_segmentation = labels['semantic_segmentation']
        assert type(semantic_segmentation) == torch.Tensor
        assert len(semantic_segmentation.shape) == 2
        assert semantic_segmentation.dtype == torch.int64
        assert set(semantic_segmentation.unique().tolist()).issubset(set(list(dataset.class_map.values()) + [dataset.IGNORE_INDEX]))
        instance_segmentation = labels['instance_segmentation']
        assert type(instance_segmentation) == torch.Tensor
        assert len(instance_segmentation.shape) == 3 and instance_segmentation.shape[0] == 2
        assert instance_segmentation.dtype == torch.float32
        # inspect meta info
        meta_info = example['meta_info']
        assert type(meta_info) == dict
        assert set(meta_info.keys()) == set(['image_filepath', 'image_resolution'])
        image_filepath = meta_info['image_filepath']
        assert type(image_filepath) == str
        assert os.path.isfile(os.path.join(dataset.data_root, image_filepath))
        image_resolution = meta_info['image_resolution']
        assert type(image_resolution) == tuple
        assert len(image_resolution) == 2
        assert image.shape[-2:] == image_resolution
        assert depth_estimation.shape[-2:] == image_resolution
        assert semantic_segmentation.shape[-2:] == image_resolution
        assert instance_segmentation.shape[-2:] == image_resolution
