from typing import Tuple, List, Dict, Any
import os
import scipy
import torch
from data.datasets import BaseDataset
from utils.io import load_image


class NYUv2Dataset(BaseDataset):
    __doc__ = r"""Reference: https://github.com/facebookresearch/astmt/blob/master/fblib/dataloaders/nyud.py

    Download: https://data.vision.ee.ethz.ch/kmaninis/share/MTL/NYUD_MT.tgz

    Used in:
        GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks (https://arxiv.org/pdf/1711.02257.pdf)
        Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning (https://arxiv.org/pdf/2111.10603.pdf)
        Conflict-Averse Gradient Descent for Multi-task Learning (https://arxiv.org/pdf/2110.14048.pdf)
        FAMO: Fast Adaptive Multitask Optimization (https://arxiv.org/pdf/2306.03792.pdf)
        Towards Impartial Multi-task Learning (https://openreview.net/pdf?id=IMPnRXEWpvr)
        Multi-Task Learning as a Bargaining Game (https://arxiv.org/pdf/2202.01017.pdf)
        Independent Component Alignment for Multi-Task Learning (https://arxiv.org/pdf/2305.19000.pdf)
        Regularizing Deep Multi-Task Networks using Orthogonal Gradients (https://arxiv.org/pdf/1912.06844.pdf)
        Gradient Surgery for Multi-Task Learning (https://arxiv.org/pdf/2001.06782.pdf)
        Achievement-based Training Progress Balancing for Multi-Task Learning (https://openaccess.thecvf.com/content/ICCV2023/papers/Yun_Achievement-Based_Training_Progress_Balancing_for_Multi-Task_Learning_ICCV_2023_paper.pdf)
    """

    SPLIT_OPTIONS = ['train', 'val']
    DATASET_SIZE = {
        'train': 795,
        'val': 654,
    }
    INPUT_NAMES = ['image']
    LABEL_NAMES = ['depth_estimation', 'normal_estimation', 'semantic_segmentation']
    SHA1SUM = "5cd337198ead0768975610a135e26257153198c7"

    IGNORE_INDEX = 250
    CLASS_MAP_F = dict(zip(range(41), range(41)))
    CLASS_MAP_C = dict(zip(range(41), [0, 12, 5, 6, 1, 4, 9, 10, 12, 13, 6, 8, 6, 13, 10, 6, 13, 6, 7, 7, 5, 7, 3, 2, 6, 11, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 6, 7]))
    NUM_CLASSES_F = 40 + 1
    NUM_CLASSES_C = 13 + 1

    ####################################################################################################
    ####################################################################################################

    def __init__(self, semantic_granularity: str, *args, **kwargs) -> None:
        assert type(semantic_granularity) == str, f"{type(semantic_granularity)=}"
        assert semantic_granularity in ['fine', 'coarse'], f"{semantic_granularity=}"
        if semantic_granularity == 'fine':
            self.CLASS_MAP = self.CLASS_MAP_F
            self.NUM_CLASSES = self.NUM_CLASSES_F
        else:
            self.CLASS_MAP = self.CLASS_MAP_C
            self.NUM_CLASSES = self.NUM_CLASSES_C
        super(NYUv2Dataset, self).__init__(*args, **kwargs)

    def _init_annotations_(self, split: str) -> None:
        # initialize image filepaths
        image_filepaths: List[str] = []
        with open(os.path.join(os.path.join(self.data_root, "gt_sets", split + '.txt')), 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                image_fp = os.path.join(self.data_root, "images", line + '.jpg')
                assert os.path.isfile(image_fp), f"{image_fp=}"
                image_filepaths.append(image_fp)
        # initialize labels
        depth_filepaths: List[str] = []
        normal_filepaths: List[str] = []
        semantic_filepaths: List[str] = []
        edge_filepaths: List[str] = []
        for image_fp in image_filepaths:
            name = os.path.basename(image_fp).split('.')[0]
            # depth estimation
            depth_fp = os.path.join(self.data_root, "depth", name + '.mat')
            assert os.path.isfile(depth_fp), f"{depth_fp=}"
            depth_filepaths.append(depth_fp)
            # normal estimation
            normal_fp = os.path.join(self.data_root, "normals", name + '.jpg')
            assert os.path.isfile(normal_fp), f"{normal_fp=}"
            normal_filepaths.append(normal_fp)
            # semantic segmentation
            semantic_fp = os.path.join(self.data_root, "segmentation", name + '.mat')
            assert os.path.isfile(semantic_fp), f"{semantic_fp=}"
            semantic_filepaths.append(semantic_fp)
            # edge detection
            edge_fp = os.path.join(self.data_root, "edge", name + '.png')
            assert os.path.isfile(edge_fp), f"{edge_fp=}"
            edge_filepaths.append(edge_fp)
        assert len(image_filepaths) == len(edge_filepaths) == len(semantic_filepaths) == len(normal_filepaths) == len(depth_filepaths)
        # construct annotations
        self.annotations: List[Dict[str, Any]] = [{
            'image': image_filepaths[idx],
            'edge': edge_filepaths[idx],
            'semantic': semantic_filepaths[idx],
            'normal': normal_filepaths[idx],
            'depth': depth_filepaths[idx]
        } for idx in range(len(image_filepaths))]

    ####################################################################################################
    ####################################################################################################

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        inputs = self._get_image_(idx)
        labels = {}
        labels.update(self._get_depth_label_(idx))
        labels.update(self._get_normal_label_(idx))
        labels.update(self._get_segmentation_label_(idx))
        # labels.update(self._get_edge_label_(idx))
        meta_info = {
            'image_filepath': os.path.relpath(path=self.annotations[idx]['image'], start=self.data_root),
            'image_resolution': tuple(inputs['image'].shape[-2:]),
        }
        return inputs, labels, meta_info

    ####################################################################################################
    ####################################################################################################

    def _get_image_(self, idx: int) -> torch.Tensor:
        return {'image': load_image(
            filepath=self.annotations[idx]['image'], dtype=torch.float32,
            sub=None, div=255.,
        )}

    def _get_depth_label_(self, idx: int) -> Dict[str, torch.Tensor]:
        depth = torch.tensor(scipy.io.loadmat(self.annotations[idx]['depth'])['depth'], dtype=torch.float32)
        return {'depth_estimation': depth}

    def _get_normal_label_(self, idx: int) -> Dict[str, torch.Tensor]:
        normal = load_image(
            filepath=self.annotations[idx]['normal'], dtype=torch.float32,
            sub=None, div=255.,
        )
        normal = normal * 2 - 1
        return {'normal_estimation': normal}

    def _get_segmentation_label_(self, idx: int) -> Dict[str, torch.Tensor]:
        semantic = torch.tensor(data=scipy.io.loadmat(self.annotations[idx]['semantic'])['segmentation'], dtype=torch.int64)
        for class_idx in self.CLASS_MAP:
            semantic[semantic == class_idx] = self.CLASS_MAP[class_idx]
        return {'semantic_segmentation': semantic}

    def _get_edge_label_(self, idx: int) -> Dict[str, torch.Tensor]:
        edge = load_image(
            filepath=self.annotations[idx]['edge'], dtype=torch.float32,
            sub=None, div=255.,
        )
        edge = edge.unsqueeze(0)
        return {'edge_detection': edge}
