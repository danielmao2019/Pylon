from typing import Tuple, List, Dict, Any
import os
import torch
from data.datasets import BaseDataset
import utils
import itertools

class CDDDataset(BaseDataset):
    """
    The CDDDataset class is designed for managing and loading data for change detection tasks.

    References:
        * https://github.com/ServiceNow/seasonal-contrast/blob/main/datasets/oscd_dataset.py
        * https://github.com/granularai/fabric/blob/igarss2019/utils/dataloaders.py
        * https://github.com/NIX369/UNet_LSTM/blob/master/custom.py
        * https://github.com/mpapadomanolaki/UNetLSTM/blob/master/custom.py
        * https://github.com/WennyXY/DINO-MC/blob/main/data_process/oscd_dataset.py

    Download Instructions:
        1. Download the dataset from:
            https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9/edit
        2. Extract the dataset:
            ```bash
            mkdir <data-root>
            cd <data-root>
            # Download the zip files and extract them
            unrar x ChangeDetectionDataset.rar
            ```
        3. Create a soft link:
            ```bash
            ln -s <data_root_path> <Pylon_path>/data/datasets/soft_links/CDD
            ```
        4. Verify the soft link:
            ```bash
            stat <Pylon_path>/data/datasets/soft_links/CDD
            ```

    Attributes:
        SPLIT_OPTIONS (List[str]): Available data splits ('train', 'test', 'val').
        DATASET_SIZE (Optional[int]): Dataset size (set dynamically).
        INPUT_NAMES (List[str]): Names of input tensors.
        LABEL_NAMES (List[str]): Names of label tensors.
        SHA1SUM (Optional[str]): Checksum for data validation.
    """

    SPLIT_OPTIONS = ['train', 'test', 'val']
    DATASET_SIZE = None
    INPUT_NAMES = ['img_1', 'img_2']
    LABEL_NAMES = ['change_map']
    SHA1SUM = None

    def _init_annotations(self) -> None:
        subfolders = os.listdir(self.data_root)
        model_paths = [os.path.join(self.data_root, subfolder) for subfolder in subfolders]
        directories = []
        for model_path in model_paths:
            subfolders = os.listdir(model_path)
            directories = directories + [os.path.join(model_path, subfolder) for subfolder in subfolders if subfolder != 'original']

        print(directories)
        self.annotations: List[dict] = []
        for directory in directories:
            input_1_files = []
            input_2_files = []
            label_files = []
            if self.split in os.listdir(directory):
                folder_root = os.path.join(directory, self.split)
                input_1_files = [os.path.join(folder_root, 'A', filename) for filename in sorted(os.listdir(os.path.join(folder_root, 'A')),
                                                                key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))]
                input_2_files = [os.path.join(folder_root, 'B', filename) for filename in sorted(os.listdir(os.path.join(folder_root, 'B')),
                                                                key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))]
                label_files = [os.path.join(folder_root, 'OUT', filename) for filename in sorted(os.listdir(os.path.join(folder_root, 'OUT')),
                                                                key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))]
                
            for input_1_filepath, input_2_filepath, label_filepath in zip(input_1_files, input_2_files, label_files):
                # Ensure required files exist
                assert os.path.isfile(input_1_filepath), f"File not found: {input_1_filepath}"
                assert os.path.isfile(input_2_filepath), f"File not found: {input_2_filepath}"
                assert os.path.isfile(label_filepath), f"File not found: {label_filepath}"

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
        inputs = self._load_inputs(idx)
        labels = self._load_labels(idx)
        if labels['change_map'].shape[0] == 3:
            ndim, height, width = labels['change_map'].shape
        else:
            height, width = labels['change_map'].shape
        assert all(
            x.shape == (3, height, width) for x in [inputs['img_1'], inputs['img_2']]
        ), f"Shape mismatch: {inputs['img_1'].shape}, {inputs['img_2'].shape}, {labels['change_map'].shape}"

        meta_info = {'image_resolution': (height, width)}
        return inputs, labels, meta_info

    def _load_inputs(self, idx: int) -> Dict[str, torch.Tensor]:
        inputs = {}
        return {
            f'img_{i}': utils.io.load_image(
                filepath=self.annotations[idx]['inputs'][f'input_{i}_filepath'],
                dtype=torch.float32, sub=None, div=255.0,
            )
            for i in [1, 2]
        }

    def _load_labels(self, idx: int) -> Dict[str, torch.Tensor]:
        change_map = utils.io.load_image(
            filepath=self.annotations[idx]['labels']['label_filepath'],
            dtype=torch.int64, sub=None, div=255,
        )
        assert change_map.ndim == 2, f"Expected 2D label, got {change_map.shape}."
        return {'change_map': change_map}
