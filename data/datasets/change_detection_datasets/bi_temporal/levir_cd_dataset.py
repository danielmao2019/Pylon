from typing import Tuple, List, Dict, Any
import os
import torch
from data.datasets import BaseDataset
import utils


class LevirCdDataset(BaseDataset):
    """
    Dataset class for LEVIR Change Detection Dataset.

    This class handles loading, preprocessing, and annotation of the LEVIR dataset
    for change detection tasks.

    Attributes:
        SPLIT_OPTIONS (list): Available data splits ['train', 'test', 'val'].
        DATASET_SIZE (dict): Number of samples in each split.
        INPUT_NAMES (list): Names of input image pairs.
        LABEL_NAMES (list): Names of labels.
        SHA1SUM (str): SHA1 checksum for dataset verification.
    """

    __doc__ = r"""
    References:
        * https://github.com/Z-Zheng/ChangeStar/blob/master/data/levir_cd/dataset.py
        * https://github.com/AI-Zhpp/FTN/blob/main/data/dataset_swin_levir.py
        * https://github.com/ViTAE-Transformer/MTP/blob/main/RS_Tasks_Finetune/Change_Detection/opencd/datasets/levir_cd.py
        * https://gitlab.com/sbonnefoy/siamese_net_change_detection/-/blob/main/train_fc_siam_diff.ipynb?plain=0
        * https://github.com/likyoo/open-cd/blob/main/opencd/datasets/levir_cd.py
        * https://github.com/Bobholamovic/CDLab/blob/master/src/data/levircd.py

    Download:
        * https://drive.google.com/drive/folders/1dLuzldMRmbBNKPpUkX8Z53hi6NHLrWim

        # Download the file from google drive
        mkdir <data_root_path>
        cd <data_root_path>
        # unzip the package
        unzip val.zip
        unzip test.zip
        unzip train.zip
        rm val.zip
        rm test.zip
        rm train.zip
        # create softlink
        ln -s <data_root_path> <Pylon_path>/data/datasets/soft_links/LEVIR_CD
        # verify softlink status
        stat <Pylon_path>/data/datasets/soft_links/LEVIR_CD
    Used in:
    """

    SPLIT_OPTIONS = ['train', 'val', 'test']
    DATASET_SIZE = {
        'train': 445,
        'val': 64,
        'test': 128,
    }
    INPUT_NAMES = ['img_1', 'img_2']
    LABEL_NAMES = ['change_map']
    CLASS_DIST = {
        'train': [445204032, 21412334],
        'test': [127380432, 6837335],
        'val': [64292600, 2816258],
    }
    NUM_CLASSES = 2
    SHA1SUM = '610f742580165b4af94ffae295dbab8986a92b69'


    def _init_annotations(self) -> None:
        """
        Initialize dataset annotations.

        Raises:
            AssertionError: If any expected file is missing.
        """
        inputs_root: str = os.path.join(self.data_root, f"{self.split}")
        labels_root: str = os.path.join(self.data_root, f"{self.split}", "label")
        self.annotations: List[dict] = []
        files_list = sorted(
            os.listdir(os.path.join(inputs_root, 'A')),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]),
        )
        assert len(os.listdir(os.path.join(inputs_root, 'A'))), len(os.listdir(os.path.join(inputs_root, 'B')))
        for filename in files_list:
            input_1_filepath = os.path.join(inputs_root, 'A', filename)
            assert os.path.isfile(input_1_filepath), f"{input_1_filepath=}"
            input_2_filepath = os.path.join(inputs_root, 'B', filename)
            assert os.path.isfile(input_1_filepath), f"{input_1_filepath=}"
            label_filepath = os.path.join(labels_root, filename)
            assert os.path.isfile(label_filepath), f"{label_filepath=}"
            self.annotations.append({
                'inputs': {
                    'input_1_filepath': input_1_filepath,
                    'input_2_filepath': input_2_filepath,
                },
                'labels': {
                    'label_filepath': label_filepath,
                },
            })

    def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Load a single datapoint by index.

        Args:
            idx (int): Index of the datapoint to load.

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
                A tuple containing inputs, labels, and metadata.

        Raises:
            AssertionError: If input or label dimensions mismatch.
        """
        inputs = self._load_inputs(idx)
        labels = self._load_labels(idx)
        height, width = labels['change_map'].shape
        assert all(x.shape == (3, height, width) for x in [inputs['img_1'], inputs['img_2']]), \
            f"{inputs['img_1'].shape=}, {inputs['img_2'].shape=}, {labels['change_map'].shape=}"
        meta_info = {
            'image_resolution': (height, width),
        }
        return inputs, labels, meta_info

    def _load_inputs(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load input images for a given index.

        Args:
            idx (int): Index of the datapoint.

        Returns:
            Dict[str, torch.Tensor]: Dictionary with loaded input images.
        """
        inputs: Dict[str, torch.Tensor] = {}
        for input_idx in [1, 2]:
            img = utils.io.load_image(
                filepath=self.annotations[idx]['inputs'][f'input_{input_idx}_filepath'],
                dtype=torch.float32, sub=None, div=255.0,
            )
            inputs[f'img_{input_idx}'] = img
        return inputs

    def _load_labels(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load label for a given index.

        Args:
            idx (int): Index of the datapoint.

        Returns:
            Dict[str, torch.Tensor]: Dictionary with loaded labels.

        Raises:
            AssertionError: If the loaded label is not 2D.
        """
        change_map = utils.io.load_image(
            filepath=self.annotations[idx]['labels']['label_filepath'],
            dtype=torch.int64, sub=None, div=255.0,
        )
        assert change_map.ndim == 2, f"{change_map.shape=}"
        labels = {'change_map': change_map}
        return labels
