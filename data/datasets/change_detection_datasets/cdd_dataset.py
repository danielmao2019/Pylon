from typing import Tuple, List, Dict, Any
import os
import torch
from data.datasets import BaseDataset
import utils

class CDDDataset(BaseDataset):
    __doc__ = r"""
    References:
        * https://github.com/ServiceNow/seasonal-contrast/blob/main/datasets/oscd_dataset.py
        * https://github.com/granularai/fabric/blob/igarss2019/utils/dataloaders.py
        * https://github.com/NIX369/UNet_LSTM/blob/master/custom.py
        * https://github.com/mpapadomanolaki/UNetLSTM/blob/master/custom.py
        * https://github.com/WennyXY/DINO-MC/blob/main/data_process/oscd_dataset.py

    Download:
        * https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9/edit
        ```bash
        mkdir <data-root>
        cd <data-root>
        # <download the zip files from the link above>
        unrar x ChangeDetectionDataset.rar
        # create softlink
        ln -s <data_root_path> <Pylon_path>/data/datasets/soft_links/CDD
        # verify softlink status
        stat <Pylon_path>/data/datasets/soft_links/CDD
        ```
    Used in:

    """
    
    # do a git rebase and review the dataset structure
    
    SPLIT_OPTIONS = ['train', 'test', 'val']
    DATASET_SIZE = None
    INPUT_NAMES = ['img_1', 'img_2']
    LABEL_NAMES = ['change_map']

    SHA1SUM = None
    
    def _init_annotations_(self, split: str) -> None:
        """
        Initialize dataset annotations.

        Args:
            split (str): The data split ('train', 'test', 'val').

        Raises:
            AssertionError: If any expected file is missing.
            
        Folder structure:
        +----Model
            +----with_shift
            +----without_shift
        +----Real
            +----Original(all bmp files)
            +----subset
            
        Todo:
        - first support reading images in Model
        """
        subfolders = (os.listdir(os.path.join(self.data_root, 'Model')))
        inputs_root: str = [os.path.join(self.data_root, 'Model', f"{subfolder}", f"{split}") for subfolder in subfolders]
        labels_root: str = [os.path.join(self.data_root, 'Model', f"{subfolder}",  f"{split}", 'OUT') for subfolder in subfolders]
        self.annotations: List[dict] = []
        for input_root, label_root in zip(inputs_root, labels_root):
            filenames = [files for files in os.listdir(os.path.join(input_root, 'A')) if not files.startswith('.')]
            filenames.sort()
            print(filenames)
            assert len(os.listdir(os.path.join(input_root, 'A'))), len(os.listdir(os.path.join(input_root, 'B')))
            for filename in filenames:
                input_1_filepath = os.path.join(input_root, 'A', filename)
                assert os.path.isfile(input_1_filepath), f"{input_1_filepath=}"
                input_2_filepath = os.path.join(input_root, 'B', filename)
                assert os.path.isfile(input_1_filepath), f"{input_1_filepath=}"
                label_filepath = os.path.join(label_root, filename)
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
