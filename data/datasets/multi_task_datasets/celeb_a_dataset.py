from typing import Tuple, List, Dict, Any, Optional
import os
import torch
from data.datasets import BaseDataset
import utils


class CelebADataset(BaseDataset):
    __doc__ = r"""
    CelebA dataset for multi-task learning with facial attribute classification tasks.

    For detailed documentation, see: docs/datasets/multi_task/celeba.md
    """

    TOTAL_SIZE = 202599
    SPLIT_OPTIONS = ['train', 'val', 'test']
    DATASET_SIZE = {
        'train': 162770,
        'val': 19867,
        'test': 19962,
    }
    INPUT_NAMES = ['image']
    LABEL_NAMES = ['landmarks'] + [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
        'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
        'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
        'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
        'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
        'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young',
    ]
    SHA1SUM = "5cd337198ead0768975610a135e26257153198c7"

    # ====================================================================================================
    # initialization methods
    # ====================================================================================================

    def __init__(self, use_landmarks: Optional[bool] = False, **kwargs) -> None:
        assert type(use_landmarks) == bool, f"{type(use_landmarks)=}"
        self.use_landmarks = use_landmarks
        super(CelebADataset, self).__init__(**kwargs)

    def _init_annotations(self) -> None:
        image_filepaths = self._init_images_()
        landmark_labels = self._init_landmark_labels_(image_filepaths=image_filepaths)
        attribute_labels = self._init_attribute_labels_(image_filepaths=image_filepaths)
        self.annotations = list(zip(image_filepaths, landmark_labels, attribute_labels))

    def _init_images_(self) -> List[str]:
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
                if split_enum[int(line[1])] == self.split:
                    image_filepaths.append(filepath)
        return image_filepaths

    def _init_landmark_labels_(self, image_filepaths: List[str]) -> List[torch.Tensor]:
        if not self.use_landmarks:
            return [None] * len(image_filepaths)
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
                    (name, torch.tensor((1 if val == "1" else 0), dtype=torch.int64))
                    for name, val in zip(self.LABEL_NAMES[1:], line[1:])
                )
                attribute_labels.append(attributes)
        return attribute_labels

    # ====================================================================================================
    # load methods
    # ====================================================================================================

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        inputs = {'image': utils.io.load_image(
            filepath=self.annotations[idx][0],
            dtype=torch.float32, sub=None, div=255.,
        )}
        labels = {}
        if self.use_landmarks:
            labels.update({'landmarks': self.annotations[idx][1]})
        labels.update(self.annotations[idx][2])
        meta_info = {
            'image_filepath': os.path.relpath(path=self.annotations[idx][0], start=self.data_root),
            'image_resolution': tuple(inputs['image'].shape[-2:]),
        }
        return inputs, labels, meta_info
