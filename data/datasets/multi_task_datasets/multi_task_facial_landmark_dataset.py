from typing import Tuple, List, Dict, Any
import os
import torch
from data.datasets import BaseDataset
import utils


class MultiTaskFacialLandmarkDataset(BaseDataset):
    __doc__ = r"""

    Download:
        https://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html
    
    Used in:
        Facial Landmark Detection by Deep Multi-task Learning (https://link.springer.com/chapter/10.1007/978-3-319-10599-4_7)
    """

    SPLIT_OPTIONS = ['train', 'test']
    INPUT_NAMES = ['image']
    LABEL_NAMES = ['landmarks', 'gender', 'smile', 'glasses', 'pose']

    ####################################################################################################
    ####################################################################################################

    def _init_annotations(self) -> None:
        image_filepaths = self._init_images_()
        all_labels = self._init_labels_(image_filepaths=image_filepaths)
        self.annotations = list(zip(image_filepaths, all_labels))

    def _init_images_(self) -> None:
        image_filepaths = []
        with open(os.path.join(self.data_root, f"{self.split}ing.txt"), mode='r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                filepath = os.path.join(self.data_root, line[0])
                assert os.path.isfile(filepath), f"{filepath=}"
                image_filepaths.append(filepath)
        return image_filepaths

    def _init_labels_(self, image_filepaths: List[str]) -> None:
        # image
        all_labels: List[Dict[str, torch.Tensor]] = []
        with open(os.path.join(self.data_root, f"{self.split}ing.txt"), mode='r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                line = line.strip().split()
                assert line[0] == os.path.relpath(path=image_filepaths[idx], start=self.data_root), \
                    f"{idx=}, {line[0]=}, {image_filepaths[idx]=}, {self.data_root=}"
                landmarks = torch.tensor(list(map(float, [c for coord in zip(line[1:6], line[6:11]) for c in coord])), dtype=torch.float32)
                attributes = dict(
                    (name, torch.tensor(int(val), dtype=torch.int8))
                    for name, val in zip(self.LABEL_NAMES[1:], line[11:15])
                )
                labels: Dict[str, torch.Tensor] = {}
                labels.update({'landmarks': landmarks})
                labels.update(attributes)
                all_labels.append(labels)
        return all_labels

    ####################################################################################################
    ####################################################################################################

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        inputs = {'image': utils.io.load_image(
            filepath=self.annotations[idx][0],
            dtype=torch.float32, sub=None, div=255.,
        )}
        labels = self.annotations[idx][1]
        meta_info = {
            'image_filepath': os.path.relpath(path=self.annotations[idx][0], start=self.data_root),
            'image_resolution': tuple(inputs['image'].shape[-2:]),
        }
        return inputs, labels, meta_info
