from typing import Any, Dict, Tuple
import os
import torch
from data.datasets.base_dataset import BaseDataset
from utils.io import load_image


class COCOStuff164KDataset(BaseDataset):
    __doc__ = r"""Reference:

    Download:
        Reference:
            https://github.com/xu-ji/IIC/blob/master/datasets/setup_cocostuff164k.sh
            https://github.com/xu-ji/IIC/blob/master/datasets/README.txt
        Steps:
            cd <data-root>
            wget -nc -P . http://images.cocodataset.org/zips/train2017.zip
            wget -nc -P . http://images.cocodataset.org/zips/val2017.zip
            wget -nc -P . http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip
            mkdir -p ./images
            mkdir -p ./annotations
            unzip -n ./train2017.zip -d ./images/
            unzip -n ./val2017.zip -d ./images/
            unzip -n ./stuffthingmaps_trainval2017.zip -d ./annotations/
            wget https://www.robots.ox.ac.uk/~xuji/datasets/COCOStuff164kCurated.tar.gz
            tar -xzvf COCOStuff164kCurated.tar.gz
            mv COCO/CocoStuff164k/curated .
            rmdir COCO/CocoStuff164k
            rmdir COCO
    """

    SPLIT_OPTIONS = ['train2017', 'val2017']
    DATASET_SIZE = {
        'train2017': None,
        'val2017': None,
    }
    INPUT_NAMES = ['image']
    LABEL_NAMES = ['label']

    FINE_TO_COARSE = {
        0: 9, 1: 11, 2: 11, 3: 11, 4: 11, 5: 11, 6: 11, 7: 11, 8: 11, 9: 8, 10: 8, 11: 8, 12: 8,
        13: 8, 14: 8, 15: 7, 16: 7, 17: 7, 18: 7, 19: 7, 20: 7, 21: 7, 22: 7, 23: 7, 24: 7,
        25: 6, 26: 6, 27: 6, 28: 6, 29: 6, 30: 6, 31: 6, 32: 6, 33: 10, 34: 10, 35: 10, 36: 10,
        37: 10, 38: 10, 39: 10, 40: 10, 41: 10, 42: 10, 43: 5, 44: 5, 45: 5, 46: 5, 47: 5, 48: 5,
        49: 5, 50: 5, 51: 2, 52: 2, 53: 2, 54: 2, 55: 2, 56: 2, 57: 2, 58: 2, 59: 2, 60: 2,
        61: 3, 62: 3, 63: 3, 64: 3, 65: 3, 66: 3, 67: 3, 68: 3, 69: 3, 70: 3, 71: 0, 72: 0,
        73: 0, 74: 0, 75: 0, 76: 0, 77: 1, 78: 1, 79: 1, 80: 1, 81: 1, 82: 1, 83: 4, 84: 4,
        85: 4, 86: 4, 87: 4, 88: 4, 89: 4, 90: 4, 91: 17, 92: 17, 93: 22, 94: 20, 95: 20, 96: 22,
        97: 15, 98: 25, 99: 16, 100: 13, 101: 12, 102: 12, 103: 17, 104: 17, 105: 23, 106: 15,
        107: 15, 108: 17, 109: 15, 110: 21, 111: 15, 112: 25, 113: 13, 114: 13, 115: 13, 116: 13,
        117: 13, 118: 22, 119: 26, 120: 14, 121: 14, 122: 15, 123: 22, 124: 21, 125: 21, 126: 24,
        127: 20, 128: 22, 129: 15, 130: 17, 131: 16, 132: 15, 133: 22, 134: 24, 135: 21, 136: 17,
        137: 25, 138: 16, 139: 21, 140: 17, 141: 22, 142: 16, 143: 21, 144: 21, 145: 25, 146: 21,
        147: 26, 148: 21, 149: 24, 150: 20, 151: 17, 152: 14, 153: 21, 154: 26, 155: 15, 156: 23,
        157: 20, 158: 21, 159: 24, 160: 15, 161: 24, 162: 22, 163: 25, 164: 15, 165: 20, 166: 17,
        167: 17, 168: 22, 169: 14, 170: 18, 171: 18, 172: 18, 173: 18, 174: 18, 175: 18, 176: 18,
        177: 26, 178: 26, 179: 19, 180: 19, 181: 24,
    }
    NUM_CLASSES_F = 182
    NUM_CLASSES_C = 27

    def __init__(self, semantic_granularity: str = 'coarse', *args, **kwargs) -> None:
        assert isinstance(semantic_granularity, str), f"{type(semantic_granularity)=}"
        assert semantic_granularity in ['fine', 'coarse'], f"{semantic_granularity=}"
        if semantic_granularity == 'fine':
            self.NUM_CLASSES = self.NUM_CLASSES_F
        else:
            self.NUM_CLASSES = self.NUM_CLASSES_C
        self.semantic_granularity = semantic_granularity
        super(COCOStuff164KDataset, self).__init__(*args, **kwargs)

    def _init_annotations(self) -> None:
        with open(os.path.join(self.data_root, 'curated', self.split, "Coco164kFull_Stuff_Coarse.txt" ), mode='r') as f:
            ids = [fn.rstrip() for fn in f.readlines()]
        self.annotations = [{
            'image_filepath': os.path.join(self.data_root, 'images', self.split, f"{id}.jpg"),
            'label_filepath': os.path.join(self.data_root, 'annotations', self.split, f"{id}.png"),
        } for id in ids]

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        inputs = {
            'image': load_image(
                filepath=self.annotations[idx]['image_filepath'],
                dtype=torch.float32, sub=None, div=255.0,
            ),
        }
        label = load_image(
            filepath=self.annotations[idx]['label_filepath'],
            dtype=torch.int64, sub=None, div=None,
        )
        if self.semantic_granularity == 'coarse':
            for cls in label.unique():
                label[label == cls] = self.FINE_TO_COARSE[cls]
        labels = {'label': label}
        meta_info = {
            'idx': idx,
            'image_filepath': os.path.relpath(path=self.annotations[idx]['image_filepath'], start=self.data_root),
            'label_filepath': os.path.relpath(path=self.annotations[idx]['label_filepath'], start=self.data_root),
            'image_resolution': tuple(inputs['image'].shape[-2:]),
        }
        return inputs, labels, meta_info
