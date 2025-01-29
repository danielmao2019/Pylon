from typing import Tuple, List, Dict, Any
import os
import torch
from data.datasets import BaseDataset
import utils
import glob

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

    SPLIT_OPTIONS = ['train', 'val', 'test']
    DATASET_SIZE = {
        'train': 26000,
        'val': 6998,
        'test': 7000,
    }
    INPUT_NAMES = ['img_1', 'img_2']
    LABEL_NAMES = ['change_map']
    CLASS_DIST = {
        'train': [1540513402, 163422598],
        'val': [414629785, 43991143],
        'test': [413621824, 45130176],
    }
    NUM_CLASSES = 2
    SHA1SUM = None

    def _init_annotations(self) -> None:
        get_files = lambda name: sorted(glob.glob(os.path.join(self.data_root, "**", "**", self.split, name, "*.jpg")))
        for img_1_path, img_2_path, change_map_path in zip(get_files('A'), get_files('B'), get_files('OUT')):
            assert all(os.path.basename(x) == os.path.basename(change_map_path) for x in [img_1_path, img_2_path])
            self.annotations.append({
                'img_1_path': img_1_path,
                'img_2_path': img_2_path,
                'change_map_path': change_map_path,
            })

    def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        inputs = {
            f'img_{i}': utils.io.load_image(
                filepath=self.annotations[idx][f'img_{i}_path'],
                dtype=torch.float32, sub=None, div=255.0,
            )
            for i in [1, 2]
        }
        labels = {
            'change_map': utils.io.load_image(
                filepath=self.annotations[idx]['change_map_path'],
                dtype=torch.int64, sub=None, div=255.0,
            )
        }
        height, width = labels['change_map'].shape
        assert all(
            x.shape == (3, height, width) for x in [inputs['img_1'], inputs['img_2']]
        ), f"Shape mismatch: {inputs['img_1'].shape}, {inputs['img_2'].shape}, {labels['change_map'].shape}"

        meta_info = {'image_resolution': (height, width)}
        return inputs, labels, meta_info
