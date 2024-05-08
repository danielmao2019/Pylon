from typing import Tuple, List, Dict, Any, Optional
import os
import glob
import numpy
import torch
from PIL import Image
from .base_dataset import BaseDataset
from utils.io import load_image


class CityScapesDataset(BaseDataset):
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
    class_map = dict(zip(valid_classes, range(19)))
    NUM_CLASSES = 19

    IMAGE_MEAN = numpy.array([123.675, 116.28, 103.53])
    DEPTH_STD = 2729.0680031169923
    DEPTH_MEAN = 0.0

    SPLIT_OPTIONS = ['train', 'val']
    INPUT_NAMES = ['image']
    LABEL_NAMES = ['depth_estimation', 'semantic_segmentation', 'instance_segmentation']

    REMOVE_INDICES = {
        'train': [253, 926, 931, 1849, 1946, 1993, 2051, 2054, 2778],
        'val': [284, 285, 286, 288, 299, 307, 312],
    }

    ####################################################################################################
    ####################################################################################################

    def _init_annotations_(self, split: str) -> None:
        # initialize image filepaths
        image_filepaths: List[str] = sorted(glob.glob(os.path.join(self.data_root, "leftImg8bit", split, "**", "*.png")))
        # depth estimation labels
        depth_root = os.path.join(self.data_root, "disparity", split)
        depth_paths = [os.path.join(
            depth_root, image_fp.split(os.sep)[-2], os.path.basename(image_fp)[:-15] + "disparity.png"
        ) for image_fp in image_filepaths]
        # semantic and instance segmentation labels
        segmentation_root = os.path.join(self.data_root, "gtFine", split)
        semantic_paths = [os.path.join(
            segmentation_root, image_fp.split(os.sep)[-2], os.path.basename(image_fp)[:-15] + "gtFine_labelIds.png"
        ) for image_fp in image_filepaths]
        instance_paths = [os.path.join(
            segmentation_root, image_fp.split(os.sep)[-2], os.path.basename(image_fp)[:-15] + "gtFine_instanceIds.png"
        ) for image_fp in image_filepaths]
        # construct annotations
        self.annotations: List[Dict[str, Any]] = [{
            'image': image_filepaths[idx],
            'depth': depth_paths[idx],
            'semantic': semantic_paths[idx],
            'instance': instance_paths[idx],
        } for idx in range(len(image_filepaths))]
        self._filter_dataset_(remove_indices=self.REMOVE_INDICES[split])

    def _filter_dataset_(self, remove_indices: List[int], cache: Optional[bool] = True) -> None:
        if not cache:
            remove_indices = []
            for idx in range(len(self.annotations)):
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
        self.annotations = [self.annotations[idx] for idx in range(len(self.annotations)) if idx not in remove_indices]

    ####################################################################################################
    ####################################################################################################

    def _load_example_(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        r"""
        Returns:
            Dict[str, Dict[str, Any]]: data point at index `idx` in the following format:
            inputs = {
                'image': float32 tensor of shape (3, H, W).
            }
            labels = {
                'depth_estimation': float32 tensor of shape (1, H, W).
                'semantic_segmentation': int64 tensor of shape (H, W).
                'instance_segmentation': float32 tensor of shape (2, H, W).
            }
            meta_info = {
                'image_filepath': str object for image file path.
                'image_resolution': 2-tuple object for image height and width.
            }
        """
        inputs = self._get_image_(idx)
        labels = {}
        labels.update(self._get_depth_label_(idx))
        labels.update(self._get_segmentation_labels_(idx))
        meta_info = {
            'image_filepath': os.path.relpath(path=self.annotations[idx]['image'], start=self.data_root),
            'image_resolution': tuple(inputs['image'].shape[-2:]),
        }
        return inputs, labels, meta_info

    ####################################################################################################
    ####################################################################################################

    def _get_image_(self, idx: int) -> Dict[str, torch.Tensor]:
        image = numpy.array(Image.open(self.annotations[idx]['image']), dtype=numpy.uint8)
        image = image[:, :, ::-1]
        image = image.astype(numpy.float64)
        image -= self.IMAGE_MEAN
        image /= 255.0
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).type(torch.float32)
        return {'image': image}

    def _get_depth_label_(self, idx: int) -> Dict[str, torch.Tensor]:
        depth = load_image(filepath=self.annotations[idx]['depth'], dtype=torch.float32)
        depth /= self.DEPTH_STD
        return {'depth_estimation': depth}

    def _get_segmentation_labels_(self, idx: int) -> Dict[str, torch.Tensor]:
        # get semantic segmentation labels
        semantic = load_image(filepath=self.annotations[idx]['semantic'], dtype=torch.int64)
        for void in self.void_classes:
            semantic[semantic == void] = self.IGNORE_INDEX
        for valid in self.valid_classes:
            semantic[semantic == valid] = self.class_map[valid]
        # get instance segmentation labels
        instance = load_image(filepath=self.annotations[idx]['instance'], dtype=torch.float32)
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
        # return result
        return {
            'semantic_segmentation': semantic,
            'instance_segmentation': instance,
        }
