from typing import Tuple, List, Dict, Any, Optional
import os
import glob
from datetime import datetime
import torch
from data.datasets import BaseDataset
import utils

class LevirCdDataset(BaseDataset):
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
    
    
    SPLIT_OPTIONS = ['train', 'test', 'val']
    DATASET_SIZE = {
        'train': 445,
        'test': 128,
        'validation': 64,
    }
    INPUT_NAMES = ['img_1', 'img_2']
    LABEL_NAMES = ['change_map']
    SHA1SUM = '5cd337198ead0768975610a135e26257153198c7'
    

    def __init__(self, bands: Optional[List[str]] = None, **kwargs) -> None:
        if bands is not None:
            assert type(bands) == list and all(type(x) == str for x in bands), f"{bands=}"
        self.bands = bands
        super(OSCDDataset, self).__init__(**kwargs)
        
    def _init_annotations_(self, split: str) -> None:
        inputs_root: str = os.path.join(self.data_root, f"{split}")
        labels_root: str = os.path.join(self.data_root, f"{split}", "label")
        self.annotations: List[dict] = []
        for i in DATASET_SIZE[f"{split}"]:
            png_input_1_filepath = os.path.join(inputs_root, 'A', str(i) + ".png")
            assert os.path.isfile(png_input_1_filepath), f"{png_input_1_filepath=}"
            png_input_2_filepath = os.path.join(inputs_root, 'B', str(i) + ".png")
            assert os.path.isfile(png_input_1_filepath), f"{png_input_1_filepath=}"
            png_label_filepath = os.path.join(labels_root, f"{split}" + str(i) + ".png")
            assert os.path.isfile(png_label_filepath), f"{png_label_filepath=}"
            # define meta info
            png_size = utils.io.load_image(filepath=png_label_filepath).shape[-2:]
            height, width = png_size
            # add annotation
            self.annotations.append({
                'inputs': {
                    'png_input_1_filepath': png_input_1_filepath,
                    'png_input_2_filepath': png_input_2_filepath,
                },
                'labels': {
                    'png_label_filepath': png_label_filepath,
                },
                'meta_info': {
                    'height': height,
                    'width': width,
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
            img = utils.io.load_image(
                filepath=self.annotations[idx]['inputs'][f'png_input_{input_idx}_filepath'],
                dtype=torch.float32, sub=None, div=255.0,
            )
            inputs[f'img_{input_idx}'] = img
        return inputs
        
    def _load_labels(self, idx: int) -> torch.Tensor:
        change_map = utils.io.load_image(
            filepaths=self.annotations[idx]['labels']['png_label_filepaths'],
            dtype=torch.int64, sub=1, div=None,  # sub 1 to convert {1, 2} to {0, 1}
            height=self.annotations[idx]['meta_info']['height'],
            width=self.annotations[idx]['meta_info']['width'],
        )
        assert change_map.ndim == 3 and change_map.shape[0] == 1, f"{change_map.shape=}"
        change_map = change_map[0]
        assert change_map.ndim == 2, f"{change_map.shape=}"
        labels = {'change_map': change_map}
        return labels