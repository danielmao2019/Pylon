from typing import Tuple, Dict, Any
import os
import glob
import torch
from data.datasets import BaseDataset
import utils


class xView2Dataset(BaseDataset):
    __doc__ = r"""
    Download:
        * https://xview2.org/download-links
        ```bash
        mkdir <data-root>
        cd <data-root>
        # download and unzip training set
        <Download Challenge training set>
        tar -xvzf train_images_labels_targets.tar
        rm train_images_labels_targets.tar
        # download and unzip tier3 set
        <Download additional Tier3 training data>
        tar -xvzf tier3.tar
        rm tier3.tar
        # download and unzip test set
        <Download Challenge test set>
        tar -xvzf test_images_labels_targets.tar
        rm test_images_labels_targets.tar
        # download and unzip hold set
        <Download Challenge holdout set>
        tar -xvzf hold_images_labels_targets.tar
        rm hold_images_labels_targets.tar
        # create a soft-link
        ln -s <data-root> <project-root>/data/datasets/soft_links

    Used in:
        * Change is Everywhere: Single-Temporal Supervised Object Change Detection in Remote Sensing Imagery
    """

    SPLIT_OPTIONS = ['train', 'test', 'hold']
    DATASET_SIZE = {
        'train': 2799,
        'test': 933,
        'hold': 933,
    }
    INPUT_NAMES = ['img_1', 'img_2']
    LABEL_NAMES = ['lbl_1', 'lbl_2']
    SHA1SUM = None

    # ====================================================================================================
    # initialization methods
    # ====================================================================================================

    def _init_annotations(self) -> None:
        # gather filepaths
        input_filepaths = sorted(glob.glob(os.path.join(self.data_root, self.split, "images", "*.png")))
        img_1_filepaths = list(filter(
            lambda x: os.path.basename(x).endswith("pre_disaster.png"), input_filepaths,
        ))
        img_2_filepaths = list(filter(
            lambda x: os.path.basename(x).endswith("post_disaster.png"), input_filepaths,
        ))
        label_filepaths = sorted(glob.glob(os.path.join(self.data_root, self.split, "targets", "*.png")))
        lbl_1_filepaths = list(filter(
            lambda x: os.path.basename(x).endswith("pre_disaster_target.png"), label_filepaths,
        ))
        lbl_2_filepaths = list(filter(
            lambda x: os.path.basename(x).endswith("post_disaster_target.png"), label_filepaths,
        ))
        assert len(img_1_filepaths) == len(img_2_filepaths) == len(lbl_1_filepaths) == len(lbl_2_filepaths)
        # define annotations
        self.annotations = [{
            'inputs': {
                'img_1': img_1, 'img_2': img_2,
            },
            'labels': {
                'lbl_1': lbl_1, 'lbl_2': lbl_2,
            },
        } for img_1, img_2, lbl_1, lbl_2 in zip(img_1_filepaths, img_2_filepaths, lbl_1_filepaths, lbl_2_filepaths)]

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        inputs = {
            'img_1': utils.io.load_image(
                filepath=self.annotations[idx]['inputs']['img_1'],
                dtype=torch.float32, sub=0, div=255.0,
            ),
            'img_2': utils.io.load_image(
                filepath=self.annotations[idx]['inputs']['img_2'],
                dtype=torch.float32, sub=0, div=255.0,
            ),
        }
        labels = {
            'lbl_1': utils.io.load_image(
                filepath=self.annotations[idx]['labels']['lbl_1'],
                dtype=torch.int64, sub=None, div=None,
            ),
            'lbl_2': utils.io.load_image(
                filepath=self.annotations[idx]['labels']['lbl_2'],
                dtype=torch.int64, sub=None, div=None,
            ),
        }
        meta_info = {
            'img_1_filepath': self.annotations[idx]['inputs']['img_1'],
            'img_2_filepath': self.annotations[idx]['inputs']['img_2'],
            'lbl_1_filepath': self.annotations[idx]['labels']['lbl_1'],
            'lbl_2_filepath': self.annotations[idx]['labels']['lbl_2'],
        }
        return inputs, labels, meta_info
