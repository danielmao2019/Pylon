from typing import Tuple, Dict, Any
import os
import glob
import torch
from data.datasets import BaseDataset
import utils


class WHU_BD_Dataset(BaseDataset):

    SPLIT_OPTIONS = ['train', 'val', 'test']
    DATASET_SIZE = {
        'train': 4736,
        'val': 1036,
        'test': 2416,
    }
    INPUT_NAMES = ['image']
    LABEL_NAMES = ['semantic_map']
    NUM_CLASSES = 2
    SHA1SUM = "4057a1dfffd59ecd6d3ff169b8f503644b728592"

    def _init_annotations(self) -> None:
        image_filepaths = sorted(glob.glob(os.path.join(self.data_root, self.split, "image", "*.tif")))
        label_filepaths = sorted(glob.glob(os.path.join(self.data_root, self.split, "label", "*.tif")))
        assert all(os.path.basename(x) == os.path.basename(y) for x, y in zip(image_filepaths, label_filepaths))
        self.annotations = list(map(lambda x: {'image': x[0], 'label': x[1]}, zip(image_filepaths, label_filepaths)))

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        inputs = {
            'image': utils.io.load_image(
                filepaths=[self.annotations[idx]['image']],
                dtype=torch.float32, sub=None, div=255.0,
            )
        }
        labels = {
            'semantic_map': utils.io.load_image(
                filepaths=[self.annotations[idx]['label']],
                dtype=torch.int64, sub=None, div=None,
            ).squeeze(0)
        }
        meta_info = self.annotations[idx]
        return inputs, labels, meta_info
