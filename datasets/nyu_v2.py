from typing import Tuple, List, Dict, Any
import os
import numpy
import scipy
import torch
import torchvision
from PIL import Image

from .base_dataset import BaseDataset


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

    IGNORE_INDEX = 250
    NUM_CLASSES = 41
    SPLIT_OPTIONS = ['train', 'val', 'test']
    INPUT_NAMES = ['image']
    LABEL_NAMES = ['depth_estimation', 'normal_estimation', 'semantic_segmentation']

    def __init__(
        self, data_root: str, split: str, indices: List[int] = None,
        transforms: dict = None,
    ):
        super().__init__(data_root=data_root, split=split, transforms=transforms, indices=indices)

    ####################################################################################################
    ####################################################################################################

    def _init_images_(self, split: str) -> None:
        # input check
        assert type(split) == str, f"{type(split)=}"
        assert split in self.SPLIT_OPTIONS, f"{split=}, {self.SPLIT_OPTIONS=}"
        if split == 'test':
            self.image_filepaths = []
            return
        # initialize
        self.image_filepaths = []
        # define filepaths
        with open(os.path.join(os.path.join(self.data_root, "gt_sets", split + '.txt')), 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                image_fp = os.path.join(self.data_root, "images", line + '.jpg')
                assert os.path.isfile(image_fp), f"{image_fp=}"
                self.image_filepaths.append(image_fp)

    def _init_labels_(self, split: str) -> None:
        self.edge_fps = []
        self.segmentation_fps = []
        self.normals_fps = []
        self.depth_fps = []
        for image_fp in self.image_filepaths:
            name = os.path.basename(image_fp).split('.')[0]
            # edge
            edge_fp = os.path.join(self.data_root, "edge", name + '.png')
            assert os.path.isfile(edge_fp), f"{edge_fp=}"
            self.edge_fps.append(edge_fp)
            # segmentation
            segmentation_fp = os.path.join(self.data_root, "segmentation", name + '.mat')
            assert os.path.isfile(segmentation_fp), f"{segmentation_fp=}"
            self.segmentation_fps.append(segmentation_fp)
            # normals
            normals_fp = os.path.join(self.data_root, "normals", name + '.jpg')
            assert os.path.isfile(normals_fp), f"{normals_fp=}"
            self.normals_fps.append(normals_fp)
            # depth
            depth_fp = os.path.join(self.data_root, "depth", name + '.mat')
            assert os.path.isfile(depth_fp), f"{depth_fp=}"
            self.depth_fps.append(depth_fp)
        assert len(self.image_filepaths) == len(self.edge_fps) == len(self.segmentation_fps) == len(self.normals_fps) == len(self.depth_fps)

    ####################################################################################################
    ####################################################################################################

    def _load_example_(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        inputs = self._get_image_(idx)
        labels = {}
        labels.update(self._get_edge_label_(idx))
        labels.update(self._get_depth_label_(idx))
        labels.update(self._get_normals_label_(idx))
        labels.update(self._get_segmentation_label_(idx))
        meta_info = {
            'image_filepath': os.path.relpath(path=self.image_filepaths[idx], start=self.data_root),
            'image_resolution': tuple(inputs['image'].shape[-2:]),
        }
        return inputs, labels, meta_info

    ####################################################################################################
    ####################################################################################################

    def _get_image_(self, idx: int) -> torch.Tensor:
        image = torchvision.transforms.ToTensor()(Image.open(self.image_filepaths[idx]))
        return {'image': image}

    def _get_edge_label_(self, idx: int) -> Dict[str, torch.Tensor]:
        edge = torch.tensor(data=numpy.array(Image.open(self.edge_fps[idx])), dtype=torch.float32)
        edge = edge.unsqueeze(0) / 255
        return {'edge_detection': edge}

    def _get_depth_label_(self, idx: int) -> Dict[str, torch.Tensor]:
        depth = torch.tensor(scipy.io.loadmat(self.depth_fps[idx])['depth'], dtype=torch.float32)
        depth = depth.unsqueeze(0)
        return {'depth_estimation': depth}

    def _get_normals_label_(self, idx: int) -> Dict[str, torch.Tensor]:
        normals = torch.tensor(data=numpy.array(Image.open(self.normals_fps[idx])), dtype=torch.float32)
        normals = normals.permute(dims=(2, 0, 1)) / 255 * 2 - 1
        return {'normal_estimation': normals}

    def _get_segmentation_label_(self, idx: int) -> Dict[str, torch.Tensor]:
        segmentation = torch.tensor(data=scipy.io.loadmat(self.segmentation_fps[idx])['segmentation'], dtype=torch.int64)
        return {'semantic_segmentation': segmentation}
