from typing import Tuple, List, Dict, Any, Optional
import os
import glob
import random
import torch
import matplotlib.pyplot as plt
from data.datasets import BaseDataset
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

    SPLIT_OPTIONS = ['train', 'val']
    DATASET_SIZE = {
        'train': 2975 - 9,
        'val': 500 - 7,
    }
    INPUT_NAMES = ['image']
    LABEL_NAMES = ['depth_estimation', 'semantic_segmentation', 'instance_segmentation']
    SHA1SUM = "5cd337198ead0768975610a135e26257153198c7"

    IGNORE_INDEX = 250
    INSTANCE_VOID: List[int] = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23]
    SEMANTIC_VOID: List[int] = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
    VALID_CLASSES: List[int] = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    CLASS_MAP_F = dict(zip(VALID_CLASSES, range(19)))
    CLASS_MAP_C = {
        7: 0,  # flat
        8: 0,  # flat
        11: 1,  # construction
        12: 1,  # construction
        13: 1,  # construction
        17: 5,  # object
        19: 5,  # object
        20: 5,  # object
        21: 2,  # nature
        22: 2,  # nature
        23: 4,  # sky
        24: 6,  # human
        25: 6,  # human
        26: 3,  # vehicle
        27: 3,  # vehicle
        28: 3,  # vehicle
        31: 3,  # vehicle
        32: 3,  # vehicle
        33: 3,  # vehicle
    }
    NUM_CLASSES_F = 19
    NUM_CLASSES_C = 7

    IMAGE_MEAN = [123.675, 116.28, 103.53]
    DEPTH_STD = 2729.0680031169923
    DEPTH_MEAN = 0.0

    REMOVE_INDICES = {
        'train': [253, 926, 931, 1849, 1946, 1993, 2051, 2054, 2778],
        'val': [284, 285, 286, 288, 299, 307, 312],
    }

    CLASS_COLORS = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    # ====================================================================================================
    # initialization methods
    # ====================================================================================================

    def __init__(self, semantic_granularity: Optional[str] = 'coarse', *args, **kwargs) -> None:
        assert type(semantic_granularity) == str, f"{type(semantic_granularity)=}"
        assert semantic_granularity in ['fine', 'coarse'], f"{semantic_granularity=}"
        if semantic_granularity == 'fine':
            self.CLASS_MAP = self.CLASS_MAP_F
            self.NUM_CLASSES = self.NUM_CLASSES_F
        else:
            self.CLASS_MAP = self.CLASS_MAP_C
            self.NUM_CLASSES = self.NUM_CLASSES_C
        super(CityScapesDataset, self).__init__(*args, **kwargs)

    def _init_annotations(self) -> None:
        # initialize image filepaths
        image_root = os.path.join(self.data_root, "leftImg8bit", self.split)
        image_paths: List[str] = sorted(glob.glob(os.path.join(image_root, "**", "*.png")))
        image_postfix = "leftImg8bit.png"
        # depth estimation labels
        depth_root = os.path.join(self.data_root, "disparity", self.split)
        depth_paths = [os.path.join(
            depth_root, fp.split(os.sep)[-2], os.path.basename(fp)[:-len(image_postfix)] + "disparity.png"
        ) for fp in image_paths]
        # semantic and instance segmentation labels
        segmentation_root = os.path.join(self.data_root, "gtFine", self.split)
        semantic_paths = [os.path.join(
            segmentation_root, fp.split(os.sep)[-2], os.path.basename(fp)[:-len(image_postfix)] + "gtFine_labelIds.png"
        ) for fp in image_paths]
        instance_paths = [os.path.join(
            segmentation_root, fp.split(os.sep)[-2], os.path.basename(fp)[:-len(image_postfix)] + "gtFine_instanceIds.png"
        ) for fp in image_paths]
        # construct annotations
        self.annotations: List[Dict[str, str]] = [{
            'image': image_paths[idx],
            'depth': depth_paths[idx],
            'semantic': semantic_paths[idx],
            'instance': instance_paths[idx],
        } for idx in range(len(image_paths))]
        self._filter_dataset_(remove_indices=self.REMOVE_INDICES[self.split])

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

    # ====================================================================================================
    # load methods
    # ====================================================================================================

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        r"""
        Returns:
            Dict[str, Dict[str, Any]]: data point at index `idx` in the following format:
            inputs = {
                'image': float32 tensor of shape (3, H, W).
            }
            labels = {
                'depth_estimation': float32 tensor of shape (H, W).
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

    def _get_image_(self, idx: int) -> Dict[str, torch.Tensor]:
        return {'image': load_image(
            filepath=self.annotations[idx]['image'], dtype=torch.float32,
            sub=self.IMAGE_MEAN[::-1], div=255.0,
        )}

    def _get_depth_label_(self, idx: int) -> Dict[str, torch.Tensor]:
        return {'depth_estimation': load_image(
            filepath=self.annotations[idx]['depth'], dtype=torch.float32,
            sub=None, div=self.DEPTH_STD,
        )}

    def _get_segmentation_labels_(self, idx: int) -> Dict[str, torch.Tensor]:
        # get semantic segmentation labels
        semantic = load_image(filepath=self.annotations[idx]['semantic'], dtype=torch.int64)
        for void_class in self.SEMANTIC_VOID:
            semantic[semantic == void_class] = self.IGNORE_INDEX
        for valid in self.VALID_CLASSES:
            semantic[semantic == valid] = self.CLASS_MAP[valid]
        # get instance segmentation labels
        instance = load_image(filepath=self.annotations[idx]['instance'], dtype=torch.int64)
        instance[semantic == self.IGNORE_INDEX] = self.IGNORE_INDEX
        for void_class in self.INSTANCE_VOID:
            instance[instance == void_class] = self.IGNORE_INDEX
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
        instance_surrogate = torch.stack([instance_y, instance_x], dim=0)
        # return result
        return {
            'semantic_segmentation': semantic,
            'instance_segmentation': instance_surrogate,
        }

    # ====================================================================================================
    # visualization methods
    # ====================================================================================================

    def _visualize_datapoint(self, datapoint: Dict[str, Dict[str, torch.Tensor]]) -> None:
        _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
        # visualize image
        image = datapoint['inputs']['image']
        ax1.imshow(((image - image.min()) / (image.max() - image.min())).permute(1, 2, 0).cpu().numpy())
        # visualize depth
        depth = datapoint['labels']['depth_estimation']
        assert depth.ndim == 2, f"{depth.shape=}"
        ax2.imshow((depth / depth.max()).cpu().numpy())
        # visualize semantic segmentation
        semantic = datapoint['labels']['semantic_segmentation']
        assert semantic.ndim == 2, f"{semantic.shape=}"
        r = torch.zeros(size=image.shape[-2:], dtype=torch.uint8)
        g = torch.zeros(size=image.shape[-2:], dtype=torch.uint8)
        b = torch.zeros(size=image.shape[-2:], dtype=torch.uint8)
        for c in range(self.NUM_CLASSES):
            r[semantic == c] = self.CLASS_COLORS[c][0]
            g[semantic == c] = self.CLASS_COLORS[c][1]
            b[semantic == c] = self.CLASS_COLORS[c][2]
        rgb = torch.stack([r, g, b], dim=2)
        rgb = rgb.type(torch.float32) / 255
        ax3.imshow(rgb.cpu().numpy())
        # visualize instance segmentation
        instance = datapoint['labels']['instance_segmentation']
        instance = torch.linalg.norm(instance, dim=0)
        instance[instance > 1] = 0
        ax4.imshow(((instance - instance.min()) / (instance.max() - instance.min())).cpu().numpy())
        # show
        plt.show()

    def visualize(self) -> None:
        while True:
            idx = random.choice(range(len(self)))
            datapoint = self[idx]
            self._visualize_datapoint(datapoint)
