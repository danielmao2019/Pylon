from typing import Tuple, List, Dict, Any, Optional
import os
import glob
from datetime import datetime
import torch
from data.datasets import BaseDataset
import utils


class OSCDDataset(BaseDataset):
    __doc__ = r"""
    References:
        * https://github.com/ServiceNow/seasonal-contrast/blob/main/datasets/oscd_dataset.py
        * https://github.com/granularai/fabric/blob/igarss2019/utils/dataloaders.py
        * https://github.com/NIX369/UNet_LSTM/blob/master/custom.py
        * https://github.com/mpapadomanolaki/UNetLSTM/blob/master/custom.py
        * https://github.com/WennyXY/DINO-MC/blob/main/data_process/oscd_dataset.py

    Download:
        * https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection

    Used in:

    """

    SPLIT_OPTIONS = ['train', 'test']
    DATASET_SIZE = {
        'train': 14,
        'test': 10,
    }
    INPUT_NAMES = ['img_1', 'img_2']
    LABEL_NAMES = ['change_map']
    SHA1SUM = None

    # ====================================================================================================
    # initialization methods
    # ====================================================================================================

    def __init__(self, bands: Optional[List[str]] = None, **kwargs) -> None:
        if bands is not None:
            assert type(bands) == list and all(type(x) == str for x in bands), f"{bands=}"
        self.bands = bands
        super(OSCDDataset, self).__init__(**kwargs)

    def _init_annotations_(self, split: str) -> None:
        inputs_root: str = os.path.join(self.data_root, "images")
        labels_root: str = os.path.join(self.data_root, f"{split}_labels")
        # determine cities to use
        filepath = os.path.join(inputs_root, f"{split}.txt")
        with open(filepath, mode='r') as f:
            cities = f.readlines()
        assert len(cities) == 1, f"{cities=}"
        cities = cities[0].strip().split(',')
        # gather annotations
        self.annotations: List[dict] = []
        for city in cities:
            # define inputs
            bands_1_filepaths = sorted(glob.glob(os.path.join(inputs_root, city, "imgs_1", "*.tif")))
            bands_2_filepaths = sorted(glob.glob(os.path.join(inputs_root, city, "imgs_2", "*.tif")))
            rgb_1_filepath = os.path.join(inputs_root, city, "pair", "img1.png")
            assert os.path.isfile(rgb_1_filepath), f"{rgb_1_filepath=}"
            rgb_2_filepath = os.path.join(inputs_root, city, "pair", "img2.png")
            assert os.path.isfile(rgb_2_filepath), f"{rgb_2_filepath=}"
            # define labels
            bands_label = os.path.join(labels_root, city, "cm", f"{city}-cm.tif")
            assert os.path.isfile(bands_label), f"{bands_label=}"
            rgb_label = os.path.join(labels_root, city, "cm", "cm.png")
            assert os.path.isfile(rgb_label), f"{rgb_label=}"
            # define meta info
            with open(os.path.join(inputs_root, city, "dates.txt"), mode='r') as f:
                date_1, date_2 = f.readlines()
            assert date_1.startswith("date_1: ")
            date_1 = datetime.strptime(date_1.strip()[len("date_1: "):], "%y%m%d")
            assert date_2.startswith("date_2: ")
            date_2 = datetime.strptime(date_2.strip()[len("date_2: "):], "%y%m%d")
            # add annotation
            self.annotations.append({
                'inputs': {
                    'bands_1_filepaths': bands_1_filepaths,
                    'bands_2_filepaths': bands_2_filepaths,
                    'rgb_1_filepath': rgb_1_filepath,
                    'rgb_2_filepath': rgb_2_filepath,
                },
                'labels': {
                    'bands_label': bands_label,
                    'rgb_label': rgb_label,
                },
                'meta_info': {
                    'date_1': date_1,
                    'date_2': date_2,
                },
            })

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        inputs = self._load_inputs(idx)
        labels = self._load_labels(idx)
        meta_info = self.annotations[idx]['meta_info']
        return inputs, labels, meta_info

    def _load_inputs(self, idx: int) -> Dict[str, torch.Tensor]:
        inputs: Dict[str, torch.Tensor] = {}
        for input_idx in [1, 2]:
            if self.bands is None:
                img = utils.io.load_image(
                    filepath=self.annotations[idx]['inputs']['rgb_1_filepath'],
                    dtype=torch.float32, sub=None, div=255.0,
                )
            else:
                img = utils.io.load_image(filepaths=list(filter(
                    lambda x: os.path.splitext(os.path.basename(x))[0].split('_')[-1] in self.bands,
                    self.annotations[idx]['inputs']['bands_1_filepaths'],
                )), dtype=torch.float32, format='bands')
            inputs[f'img_{input_idx}'] = img
        return inputs

    def _load_labels(self, idx: int) -> torch.Tensor:
        labels = {
            'change_map': utils.io.load_image(
                filepath=self.annotations[idx]['labels']['rgb_label'],
                dtype=torch.float32, sub=None, div=255.0,
            )
        }
        return labels
