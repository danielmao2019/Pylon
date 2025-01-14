from typing import Tuple, Dict, Any, Optional
import os
import glob
import torch
from data.datasets import BaseDataset
import utils


class xView2Dataset(BaseDataset):
    __doc__ = r"""
    Download:
        * https://xview2.org/download-links
        Download from the following links:
            * Download Challenge training set
            * Download additional Tier3 training data
            * Download Challenge test set
            * Download Challenge holdout set
        ```bash
        mkdir <data-root>
        cd <data-root>
        # <download the tar files from the link above>
        # unzip and rename all packages
        tar -xvzf train_images_labels_targets.tar
        rm train_images_labels_targets.tar
        tar -xvzf tier3.tar
        rm tier3.tar
        tar -xvzf test_images_labels_targets.tar
        rm test_images_labels_targets.tar
        tar -xvzf hold_images_labels_targets.tar
        rm hold_images_labels_targets.tar
        # create a soft-link
        ln -s <data-root> <project-root>/data/datasets/soft_links

    Used in:
        * Change is Everywhere: Single-Temporal Supervised Object Change Detection in Remote Sensing Imagery
    """

    SPLIT_OPTIONS = ['train', 'test', 'hold']  # 'train' will load both train and tier3 components
    DATASET_SIZE = {
        'train': 9168,
        'test': None,
        'hold': None,
    }
    INPUT_NAMES = ['image']
    LABEL_NAMES = ['semantic_segmentation']
    SHA1SUM = "5cd337198ead0768975610a135e26257153198c7"

    # ====================================================================================================
    # initialization methods
    # ====================================================================================================

    def __init__(self, pre_or_post_disaster: Optional[str] = None, **kwargs) -> None:
        if pre_or_post_disaster is not None:
            assert isinstance(pre_or_post_disaster, str)
        self.pre_or_post_disaster = pre_or_post_disaster
        super(xView2Dataset, self).__init__(**kwargs)

    def _init_annotations_(self, split: str) -> None:
        # gather filepaths
        input_filepaths = sorted(glob.glob(os.path.join(self.data_root, split, "images", "*.png")))
        label_filepaths = sorted(glob.glob(os.path.join(self.data_root, split, "targets", "*.png")))
        if split == 'train':
            input_filepaths.extend(sorted(glob.glob(os.path.join(self.data_root, "tier3", "images", "*.png"))))
            label_filepaths.extend(sorted(glob.glob(os.path.join(self.data_root, "tier3", "targets", "*.png"))))
        # apply filtering
        if self.pre_or_post_disaster is not None:
            input_filepaths = list(filter(
                lambda x: os.path.basename(x).endswith(f"{self.pre_or_post_disaster}_disaster.png"),
                input_filepaths,
            ))
            label_filepaths = list(filter(
                lambda x: os.path.basename(x).endswith(f"{self.pre_or_post_disaster}_disaster.png"),
                label_filepaths,
            ))
        # sanity check
        assert all(
            os.path.splitext(os.path.basename(x))[0]+"_target.png" == os.path.basename(y)
            for x, y in zip(input_filepaths, label_filepaths)
        ), f"{list(zip(input_filepaths, label_filepaths))=}"
        # define annotations
        self.annotations = [{
            'input_filepath': x, 'label_filepath': y,
        } for x, y in zip(input_filepaths, label_filepaths)]

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        inputs = {
            'image': utils.io.load_image(
                filepath=self.annotations[idx]['input_filepath'],
                dtype=torch.float32, sub=0, div=255.0,
            ),
        }
        labels = {
            'semantic_segmentation': utils.io.load_image(
                filepath=self.annotations[idx]['label_filepath'],
                dtype=torch.int64, sub=None, div=None,
            ),
        }
        meta_info = {
            'input_filepath': self.annotations[idx]['input_filepath'],
            'label_filepath': self.annotations[idx]['label_filepath'],
        }
        return inputs, labels, meta_info
