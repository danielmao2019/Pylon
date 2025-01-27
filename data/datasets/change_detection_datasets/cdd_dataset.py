from typing import Tuple, List, Dict, Any
import os
import glob
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
    """

    SPLIT_OPTIONS = ['train', 'test', 'val']
    DATASET_SIZE = None
    INPUT_NAMES = ['img_1', 'img_2']
    LABEL_NAMES = ['change_map']
    SHA1SUM = None

    def _init_annotations(self) -> None:
        img_1_filepaths = sorted(glob.glob(os.path.join(self.data_root, 'Model', "**", self.split, 'A', "*.bmp")))
        img_2_filepaths = sorted(glob.glob(os.path.join(self.data_root, 'Model', "**", self.split, 'B', "*.bmp")))
        change_map_filepaths = sorted(glob.glob(os.path.join(self.data_root, 'Model', "**", self.split, 'OUT', "*.bmp")))

        self.annotations: List[dict] = []
        for img_1_path, img_2_path, change_map_path in zip(img_1_filepaths, img_2_filepaths, change_map_filepaths):
            assert all(os.path.basename(x) == os.path.basename(change_map_path) for x in [img_1_path, img_2_path])
            self.annotations.append({
                'inputs': {
                    'input_1_filepath': img_1_path,
                    'input_2_filepath': img_2_path,
                },
                'labels': {
                    'change_map_filepath': change_map_path,
                },
            })

    def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        inputs = self._load_inputs(idx)
        labels = self._load_labels(idx)
        height, width = labels['change_map'].shape

        assert all(
            x.shape == (3, height, width) for x in [inputs['img_1'], inputs['img_2']]
        ), f"Shape mismatch: {inputs['img_1'].shape}, {inputs['img_2'].shape}, {labels['change_map'].shape}"

        meta_info = {'image_resolution': (height, width)}
        return inputs, labels, meta_info

    def _load_inputs(self, idx: int) -> Dict[str, torch.Tensor]:
        inputs = {}
        for input_idx in [1, 2]:
            img = utils.io.load_image(
                filepath=self.annotations[idx]['inputs'][f'input_{input_idx}_filepath'],
                dtype=torch.float32, sub=None, div=255.0,
            )
            inputs[f'img_{input_idx}'] = img
        return inputs

    def _load_labels(self, idx: int) -> Dict[str, torch.Tensor]:
        change_map = utils.io.load_image(
            filepath=self.annotations[idx]['labels']['change_map_filepath'],
            dtype=torch.int64, sub=None, div=255.0,
        )
        assert change_map.ndim == 2, f"Expected 2D label, got {change_map.shape}."
        return {'change_map': change_map}
