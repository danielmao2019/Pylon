from typing import Tuple, Dict, Any
import os
import glob
import torch
from data.datasets.change_detection_datasets.base_2dcd_dataset import Base2DCDDataset
import utils


class xView2Dataset(Base2DCDDataset):
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
        # determine which directories to search based on split
        if self.split == 'train':
            # For train split, search in both tier1 and tier3 directories
            search_dirs = ['tier1', 'tier3']
        else:
            # For test/hold splits, use the split name directly
            search_dirs = [self.split]
        
        # gather filepaths from all relevant directories
        input_filepaths = []
        label_filepaths = []
        for search_dir in search_dirs:
            input_filepaths.extend(sorted(glob.glob(os.path.join(self.data_root, search_dir, "images", "*.png"))))
            label_filepaths.extend(sorted(glob.glob(os.path.join(self.data_root, search_dir, "targets", "*.png"))))
        
        img_1_filepaths = list(filter(
            lambda x: os.path.basename(x).endswith("pre_disaster.png"), input_filepaths,
        ))
        img_2_filepaths = list(filter(
            lambda x: os.path.basename(x).endswith("post_disaster.png"), input_filepaths,
        ))
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

    def _get_cache_version_dict(self) -> Dict[str, Any]:
        """Return parameters that affect dataset content for cache versioning."""
        version_dict = super()._get_cache_version_dict()
        # xView2Dataset uses standard loading without dataset-specific parameters
        return version_dict

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
