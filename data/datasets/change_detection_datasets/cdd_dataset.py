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
    
    SPLIT_OPTIONS = ['train', 'test', 'val']
    DATASET_SIZE = None
    INPUT_NAMES = ['img_1', 'img_2']
    LABEL_NAMES = ['change_map']

    SHA1SUM = None
    
    # question: how to impl the dataset given .bmp file
    