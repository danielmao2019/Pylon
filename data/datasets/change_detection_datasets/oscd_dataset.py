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
            tif_input_1_filepaths = sorted(glob.glob(os.path.join(inputs_root, city, "imgs_1", "*.tif")))
            tif_input_2_filepaths = sorted(glob.glob(os.path.join(inputs_root, city, "imgs_2", "*.tif")))
            png_input_1_filepath = os.path.join(inputs_root, city, "pair", "img1.png")
            assert os.path.isfile(png_input_1_filepath), f"{png_input_1_filepath=}"
            png_input_2_filepath = os.path.join(inputs_root, city, "pair", "img2.png")
            assert os.path.isfile(png_input_2_filepath), f"{png_input_2_filepath=}"
            # define labels
            tif_label_filepaths = [os.path.join(labels_root, city, "cm", f"{city}-cm.tif")]
            assert os.path.isfile(tif_label_filepaths[0]), f"{tif_label_filepaths=}"
            png_label_filepath = [os.path.join(labels_root, city, "cm", "cm.png")]
            assert os.path.isfile(png_label_filepath[0]), f"{png_label_filepath=}"
            # define meta info
            tif_size = utils.io.load_image(filepaths=tif_label_filepaths).shape[-2:]
            png_size = utils.io.load_image(filepath=png_label_filepath).shape[-2:]
            assert tif_size == png_size, f"{tif_size=}, {png_size=}"
            height, width = tif_size
            with open(os.path.join(inputs_root, city, "dates.txt"), mode='r') as f:
                date_1, date_2 = f.readlines()
            assert date_1.startswith("date_1: ")
            date_1 = datetime.strptime(date_1.strip()[len("date_1: "):], "%Y%m%d")
            assert date_2.startswith("date_2: ")
            date_2 = datetime.strptime(date_2.strip()[len("date_2: "):], "%Y%m%d")
            # add annotation
            self.annotations.append({
                'inputs': {
                    'tif_input_1_filepaths': tif_input_1_filepaths,
                    'tif_input_2_filepaths': tif_input_2_filepaths,
                    'png_input_1_filepath': png_input_1_filepath,
                    'png_input_2_filepath': png_input_2_filepath,
                },
                'labels': {
                    'tif_label_filepaths': tif_label_filepaths,
                    'png_label_filepath': png_label_filepath,
                },
                'meta_info': {
                    'height': height,
                    'width': width,
                    'date_1': date_1,
                    'date_2': date_2,
                },
            })

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        meta_info = self.annotations[idx]['meta_info']
        height, width = meta_info['height'], meta_info['width']
        inputs = self._load_inputs(idx)
        labels = self._load_labels(idx)
        assert all(x.shape[-2:] == (height, width) for x in [inputs['img_1'], inputs['img_2'], labels['change_map']]), \
            f"{inputs['img_1'].shape=}, {inputs['img_2'].shape=}, {labels['change_map'].shape=}"
        return inputs, labels, meta_info

    def _load_inputs(self, idx: int) -> Dict[str, torch.Tensor]:
        inputs: Dict[str, torch.Tensor] = {}
        for input_idx in [1, 2]:
            if self.bands is None:
                img = utils.io.load_image(
                    filepath=self.annotations[idx]['inputs'][f'png_input_{input_idx}_filepath'],
                    dtype=torch.float32, sub=None, div=255.0,
                )
            else:
                img = utils.io.load_image(
                    filepaths=list(filter(
                        lambda x: os.path.splitext(os.path.basename(x))[0].split('_')[-1] in self.bands,
                        self.annotations[idx]['inputs'][f'tif_input_{input_idx}_filepaths'],
                    )),
                    dtype=torch.float32, sub=None, div=None,
                    height=self.annotations[idx]['meta_info']['height'],
                    width=self.annotations[idx]['meta_info']['width'],
                )
            inputs[f'img_{input_idx}'] = img
        return inputs

    def _load_labels(self, idx: int) -> torch.Tensor:
        labels = {
            'change_map': utils.io.load_image(
                filepaths=self.annotations[idx]['labels']['tif_label_filepaths'],
                dtype=torch.int64, sub=1, div=None,  # sub 1 to convert {1, 2} to {0, 1}
                height=self.annotations[idx]['meta_info']['height'],
                width=self.annotations[idx]['meta_info']['width'],
            )
        }
        return labels
