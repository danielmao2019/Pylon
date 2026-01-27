import glob
import os
from typing import Any, Dict, Tuple

import torch

import utils
from data.datasets.semantic_segmentation_datasets.base_semseg_dataset import (
    BaseSemsegDataset,
)


class WHU_BD_Dataset(BaseSemsegDataset):

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
        assert isinstance(
            self.split, str
        ), f"Expected string split, got {type(self.split)}"
        image_filepaths = sorted(
            glob.glob(os.path.join(self.data_root, self.split, "image", "*.tif"))
        )
        label_filepaths = sorted(
            glob.glob(os.path.join(self.data_root, self.split, "label", "*.tif"))
        )
        assert all(
            os.path.basename(x) == os.path.basename(y)
            for x, y in zip(image_filepaths, label_filepaths, strict=True)
        )
        self.annotations = list(
            map(
                lambda x: {'image': x[0], 'label': x[1]},
                zip(image_filepaths, label_filepaths, strict=True),
            )
        )

    def _get_cache_version_dict(self) -> Dict[str, Any]:
        """Return parameters that affect dataset content for cache versioning."""
        # WHU_BD_Dataset has no additional parameters beyond BaseDataset
        return super()._get_cache_version_dict()

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, Any],
    ]:
        inputs = {
            'image': utils.io.image.load_image(
                filepaths=[self.annotations[idx]['image']],
                dtype=torch.float32,
                sub=None,
                div=255.0,
            )
        }
        labels = {
            'semantic_map': utils.io.image.load_image(
                filepaths=[self.annotations[idx]['label']],
                dtype=torch.int64,
                sub=None,
                div=None,
            ).squeeze(0)
        }
        # Return a copy of the annotation to avoid modifying the original
        meta_info = self.annotations[idx].copy()
        return inputs, labels, meta_info
