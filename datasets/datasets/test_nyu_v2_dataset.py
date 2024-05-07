import pytest
from .nyu_v2_dataset import NYUv2Dataset
import os
import torch


@pytest.mark.parametrize("dataset", [
    (NYUv2Dataset(data_root="./datasets/datasets/soft_links/NYUD_MT", split='train')),
    (NYUv2Dataset(data_root="./datasets/datasets/soft_links/NYUD_MT", split='train', indices=[0, 2, 4, 6, 8])),
])
def test_nyu_v2(dataset: torch.utils.data.Dataset) -> None:
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
        assert set(labels.keys()) == set(['edge_detection', 'depth_estimation', 'normal_estimation', 'semantic_segmentation'])
        edge_detection = labels['edge_detection']
        assert type(edge_detection) == torch.Tensor
        assert len(edge_detection.shape) == 3 and edge_detection.shape[0] == 1
        assert edge_detection.dtype == torch.float32
        assert set(edge_detection.unique().tolist()) == set([0, 1])
        depth_estimation = labels['depth_estimation']
        assert type(depth_estimation) == torch.Tensor
        assert len(depth_estimation.shape) == 2
        assert depth_estimation.dtype == torch.float32
        normal_estimation = labels['normal_estimation']
        assert type(normal_estimation) == torch.Tensor
        assert len(normal_estimation.shape) == 3 and normal_estimation.shape[0] == 3
        assert normal_estimation.dtype == torch.float32
        semantic_segmentation = labels['semantic_segmentation']
        assert type(semantic_segmentation) == torch.Tensor
        assert len(semantic_segmentation.shape) == 2
        assert semantic_segmentation.dtype == torch.int64
        assert set(semantic_segmentation.unique().tolist()).issubset(set(range(NYUv2Dataset.NUM_CLASSES)))
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
        assert edge_detection.shape[-2:] == image_resolution
        assert depth_estimation.shape[-2:] == image_resolution
        assert normal_estimation.shape[-2:] == image_resolution
        assert semantic_segmentation.shape[-2:] == image_resolution
