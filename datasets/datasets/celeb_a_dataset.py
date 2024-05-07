from typing import Tuple, List, Dict, Any
import os
import torch
from .base_dataset import BaseDataset
from utils.io import load_image


class CelebADataset(BaseDataset):
    __doc__ = r"""

    Download: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

    Used in:
        Just Pick a Sign: Optimizing Deep Multitask Models with Gradient Sign Dropout (https://arxiv.org/pdf/2010.06808.pdf)
        Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning (https://arxiv.org/pdf/2111.10603.pdf)
        FAMO: Fast Adaptive Multitask Optimization (https://arxiv.org/pdf/2306.03792.pdf)
        Towards Impartial Multi-task Learning (https://openreview.net/pdf?id=IMPnRXEWpvr)
        Multi-Task Learning as Multi-Objective Optimization (https://arxiv.org/pdf/1810.04650.pdf)
        Gradient Surgery for Multi-Task Learning (https://arxiv.org/pdf/2001.06782.pdf)
        Heterogeneous Face Attribute Estimation: A Deep Multi-Task Learning Approach (https://arxiv.org/pdf/1706.00906.pdf)
    """

    TOTAL_SIZE = 202599
    SPLIT_OPTIONS = ['train', 'val', 'test']
    INPUT_NAMES = ['image']
    LABEL_NAMES = ['landmarks', '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                   'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

    ####################################################################################################
    ####################################################################################################

    def _init_annotations_(self, split: str) -> None:
        image_filepaths = self._init_images_(split=split)
        landmark_labels = self._init_landmark_labels_(image_filepaths=image_filepaths)
        attribute_labels = self._init_attribute_labels_(image_filepaths=image_filepaths)
        self.annotations = list(zip(image_filepaths, landmark_labels, attribute_labels))

    def _init_images_(self, split: str) -> List[str]:
        # initialize
        split_enum = {0: 'train', 1: 'val', 2: 'test'}
        # images
        images_root = os.path.join(self.data_root, "images", "img_align_celeba")
        image_filepaths: List[str] = []
        with open(os.path.join(self.data_root, "list_eval_partition.txt"), mode='r') as f:
            lines = f.readlines()
            assert len(lines) == self.TOTAL_SIZE
            for idx in range(self.TOTAL_SIZE):
                line = lines[idx].strip().split()
                assert int(line[0].split('.')[0]) == idx + 1, f"{line[0]=}, {idx=}"
                filepath = os.path.join(images_root, line[0])
                assert os.path.isfile(filepath)
                if split_enum[int(line[1])] == split:
                    image_filepaths.append(filepath)
        return image_filepaths

    def _init_landmark_labels_(self, image_filepaths: List[str]) -> List[torch.Tensor]:
        with open(os.path.join(self.data_root, "list_landmarks_align_celeba.txt"), mode='r') as f:
            lines = f.readlines()
            assert len(lines[0].strip().split()) == 10, f"{lines[0].strip().split()=}"
            lines = lines[1:]
            assert len(lines) == self.TOTAL_SIZE
            landmark_labels: List[torch.Tensor] = []
            for fp in image_filepaths:
                idx = int(os.path.basename(fp).split('.')[0]) - 1
                line = lines[idx].strip().split()
                assert int(line[0].split('.')[0]) == idx + 1, f"{fp=}, {line[0]=}, {idx=}"
                landmarks = torch.tensor(list(map(int, line[1:])), dtype=torch.uint8)
                assert landmarks.shape == (10,), f"{landmarks.shape=}"
                landmark_labels.append(landmarks)
        return landmark_labels

    def _init_attribute_labels_(self, image_filepaths: List[str]) -> List[Dict[str, torch.Tensor]]:
        with open(os.path.join(self.data_root, "list_attr_celeba.txt"), mode='r') as f:
            lines = f.readlines()
            assert set(lines[0].strip().split()) == set(self.LABEL_NAMES[1:])
            lines = lines[1:]
            assert len(lines) == self.TOTAL_SIZE
            attribute_labels: List[Dict[str, torch.Tensor]] = []
            for fp in image_filepaths:
                idx = int(os.path.basename(fp).split('.')[0]) - 1
                line = lines[idx].strip().split()
                assert int(line[0].split('.')[0]) == idx + 1, f"{fp=}, {line[0]=}, {idx=}"
                attributes: Dict[str, torch.Tensor] = dict(
                    (name, torch.tensor([1 if val == "1" else 0], dtype=torch.int8))
                    for name, val in zip(self.LABEL_NAMES[1:], line[1:])
                )
                attribute_labels.append(attributes)
        return attribute_labels

    ####################################################################################################
    ####################################################################################################

    def _load_example_(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        inputs = {'image': load_image(filepath=self.annotations[idx][0], dtype=torch.float32)}
        labels = {'landmarks': self.annotations[idx][1]}
        labels.update(self.annotations[idx][2])
        meta_info = {
            'image_filepath': os.path.relpath(path=self.annotations[idx][0], start=self.data_root),
            'image_resolution': tuple(inputs['image'].shape[-2:]),
        }
        return inputs, labels, meta_info
