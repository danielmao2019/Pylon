from typing import Tuple, List, Dict, Callable
import os
import glob
import numpy
import torch
from PIL import Image

import datasets
from .base_dataset import BaseDataset


class CityScapes(BaseDataset):
    __doc__ = r"""Reference: https://github.com/SamsungLabs/MTL/blob/master/code/data/datasets/cityscapes.py

    Download:
        images: https://www.cityscapes-dataset.com/file-handling/?packageID=3
        segmentation labels: https://www.cityscapes-dataset.com/file-handling/?packageID=1
        disparity labels: https://www.cityscapes-dataset.com/file-handling/?packageID=7

    Used in:
        Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics (https://arxiv.org/pdf/1705.07115.pdf)
        Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning (https://arxiv.org/pdf/2111.10603.pdf)
        Conflict-Averse Gradient Descent for Multi-task Learning (https://arxiv.org/pdf/2110.14048.pdf)
        FAMO: Fast Adaptive Multitask Optimization (https://arxiv.org/pdf/2306.03792.pdf)
        Towards Impartial Multi-task Learning (https://openreview.net/pdf?id=IMPnRXEWpvr)
        Multi-Task Learning as a Bargaining Game (https://arxiv.org/pdf/2202.01017.pdf)
        Multi-Task Learning as Multi-Objective Optimization (https://arxiv.org/pdf/1810.04650.pdf)
        Independent Component Alignment for Multi-Task Learning (https://arxiv.org/pdf/2305.19000.pdf)
        Gradient Surgery for Multi-Task Learning (https://arxiv.org/pdf/2001.06782.pdf)
    """

    IGNORE_INDEX = 250
    no_instances = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23]
    void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
    valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    NUM_CLASSES = 19

    IMAGE_MEAN = numpy.array([123.675, 116.28, 103.53])
    DEPTH_STD = 2729.0680031169923
    DEPTH_MEAN = 0.0

    SPLIT_OPTIONS = ['train', 'val', 'test']
    TASK_NAMES = ['depth_estimation', 'semantic_segmentation', 'instance_segmentation']

    REMOVE_INDICES = {
        'train': [253, 926, 931, 1849, 1946, 1993, 2051, 2054, 2778],
        'val': [284, 285, 286, 288, 299, 307, 312],
        'test': None,
    }

    def __init__(
        self, data_root: str, split: str, indices: List[int] = None,
        transforms: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(data_root=data_root, split=split, transforms=transforms, indices=indices)
        self.class_map = dict(zip(self.valid_classes, range(19)))
        self._filter_dataset_(remove_indices=self.REMOVE_INDICES[split])

    ####################################################################################################
    ####################################################################################################

    def _init_images_(self, split: str) -> None:
        assert type(split) == str, f"{type(split)=}"
        assert split in self.SPLIT_OPTIONS, f"{split=}, {self.SPLIT_OPTIONS=}"
        self.image_root = os.path.join(self.data_root, "leftImg8bit", split)
        self.image_filepaths = sorted(glob.glob(os.path.join(self.image_root, "**", "*.png")))
        if split == 'test':
            self.image_filepaths = []

    def _init_labels_(self, split: str) -> None:
        # input check
        assert type(split) == str, f"{type(split)=}"
        assert split in self.SPLIT_OPTIONS, f"{split=}, {self.SPLIT_OPTIONS=}"
        # depth estimation labels
        self.depth_root = os.path.join(self.data_root, "disparity", split)
        self.depth_paths = [os.path.join(self.depth_root, image_fp.split(os.sep)[-2], os.path.basename(image_fp)[:-15] + "disparity.png")
                            for image_fp in self.image_filepaths]
        # segmentation labels
        self.segmentation_root = os.path.join(self.data_root, "gtFine", split)
        self.semantic_paths = [os.path.join(self.segmentation_root, image_fp.split(os.sep)[-2], os.path.basename(image_fp)[:-15] + "gtFine_labelIds.png")
                               for image_fp in self.image_filepaths]
        self.instance_paths = [os.path.join(self.segmentation_root, image_fp.split(os.sep)[-2], os.path.basename(image_fp)[:-15] + "gtFine_instanceIds.png")
                               for image_fp in self.image_filepaths]

    def _filter_dataset_(self, remove_indices: List[int], cache: bool = True) -> None:
        if not cache:
            remove_indices = []
            for idx in range(len(self.image_filepaths)):
                depth_estimation = self._get_depth_label_(idx)['depth_estimation']
                segmentation = self._get_segmentation_labels_(idx)
                semantic_segmentation = segmentation['semantic_segmentation']
                instance_segmentation = segmentation['instance_segmentation']
                if torch.all(depth_estimation == 0):
                    remove_indices.append(idx)
                    continue
                if torch.all(semantic_segmentation == self.IGNORE_INDEX):
                    remove_indices.append(idx)
                    continue
                if torch.all(instance_segmentation == self.IGNORE_INDEX):
                    remove_indices.append(idx)
                    continue
        self.image_filepaths = [self.image_filepaths[idx] for idx in range(len(self.image_filepaths)) if idx not in remove_indices]
        self.depth_paths = [self.depth_paths[idx] for idx in range(len(self.depth_paths)) if idx not in remove_indices]
        self.semantic_paths = [self.semantic_paths[idx] for idx in range(len(self.semantic_paths)) if idx not in remove_indices]
        self.instance_paths = [self.instance_paths[idx] for idx in range(len(self.instance_paths)) if idx not in remove_indices]

    ####################################################################################################
    ####################################################################################################

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r"""
        Returns:
            result (Dict[str, torch.Tensor]): a data point dictionary in format
                {
                    'image': float32 tensor of shape (3, H, W).
                    'depth_estimation': float32 tensor of shape (1, H, W).
                    'semantic_segmentation': int64 tensor of shape (H, W).
                    'instance_segmentation': float32 tensor of shape (2, H, W).
                }
        """
        idx = self.indices[idx] if self.indices is not None else idx
        result = {}
        result.update(self._get_image_(idx))
        result.update(self._get_depth_label_(idx))
        result.update(self._get_segmentation_labels_(idx))
        # sanity check
        for key in result:
            assert result['image'].shape[-2:] == result[key].shape[-2:], \
                f"{key=}, {result['image'].shape=}, {result[key].shape=}"
        # apply transforms
        result = datasets.utils.apply_transforms(transforms=self.transforms, example=result)
        return result

    ####################################################################################################
    ####################################################################################################

    def _get_image_(self, idx: int) -> torch.Tensor:
        image = numpy.array(Image.open(self.image_filepaths[idx]), dtype=numpy.uint8)
        image = image[:, :, ::-1]
        image = image.astype(numpy.float64)
        image -= self.IMAGE_MEAN
        image /= 255.0
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).type(torch.float32)
        assert len(image.shape) == 3 and image.shape[0] == 3, f"{image.shape=}"
        return {'image': image}

    def _get_depth_label_(self, idx: int) -> Dict[str, torch.Tensor]:
        depth = torch.tensor(data=numpy.array(Image.open(self.depth_paths[idx])), dtype=torch.float32)
        depth = depth.unsqueeze(0)
        depth /= self.DEPTH_STD
        assert len(depth.shape) == 3 and depth.shape[0] == 1, f"{depth.shape=}"
        return {'depth_estimation': depth}

    def _get_segmentation_labels_(self, idx: int) -> Dict[str, torch.Tensor]:
        # get semantic segmentation labels
        semantic = torch.tensor(data=numpy.array(Image.open(self.semantic_paths[idx])), dtype=torch.int64)
        for void in self.void_classes:
            semantic[semantic == void] = self.IGNORE_INDEX
        for valid in self.valid_classes:
            semantic[semantic == valid] = self.class_map[valid]
        # get instance segmentation labels
        instance = torch.tensor(data=numpy.array(Image.open(self.instance_paths[idx])), dtype=torch.float32)
        instance[semantic == self.IGNORE_INDEX] = self.IGNORE_INDEX
        for _no_instance in self.no_instances:
            instance[instance == _no_instance] = self.IGNORE_INDEX
        instance[instance == 0] = self.IGNORE_INDEX
        assert len(instance.shape) == 2, f"{instance.shape=}"
        height, width = instance.shape
        ymap, xmap = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        assert ymap.max() == height - 1, f"{ymap.max()=}, {height=}"
        assert xmap.max() == width - 1, f"{xmap.max()=}, {width=}"
        ymap = ymap.type(torch.float32) / ymap.max()
        xmap = xmap.type(torch.float32) / xmap.max()
        instance_y = torch.ones_like(instance, dtype=torch.float32) * self.IGNORE_INDEX
        instance_x = torch.ones_like(instance, dtype=torch.float32) * self.IGNORE_INDEX
        assert instance_y.shape == ymap.shape == instance_x.shape == xmap.shape
        for instance_id in torch.unique(instance):
            if instance_id == self.IGNORE_INDEX:
                continue
            mask = instance == instance_id
            instance_y[mask] = ymap[mask] - torch.mean(ymap[mask])
            instance_x[mask] = xmap[mask] - torch.mean(xmap[mask])
        instance = torch.stack([instance_y, instance_x], dim=0)
        # output check
        assert len(semantic.shape) == 2, f"{semantic.shape=}"
        assert semantic.dtype == torch.int64, f"{semantic.dtype=}"
        assert set(semantic.unique().tolist()).issubset(set(list(self.class_map.values()) + [self.IGNORE_INDEX])), \
            f"{set(semantic.unique().tolist())}, {set(list(self.class_map.values()) + [self.IGNORE_INDEX])=}"
        assert len(instance.shape) == 3 and instance.shape[0] == 2, f"{instance.shape=}"
        assert instance.dtype == torch.float32, f"{instance.shape=}"
        # return result
        return {
            'semantic_segmentation': semantic,
            'instance_segmentation': instance,
        }
