from typing import Tuple, Dict, Any
import os
import glob
import torch
from data.datasets import BaseDataset
import utils


class SYSU_CD_Dataset(BaseDataset):
    __doc__ = r"""
    References:
        * https://github.com/liumency/SYSU-CD

    Download:
        * https://mail2sysueducn-my.sharepoint.com/:f:/g/personal/liumx23_mail2_sysu_edu_cn/Emgc0jtEcshAnRkgq1ZTE9AB-kfXzSEzU_PAQ-5YF8Neaw?e=IhVeeZ
    """

    SPLIT_OPTIONS = ['train', 'val', 'test']
    DATASET_SIZE = {
        'train': 12000,
        'val': 4000,
        'test': 4000,
    }
    INPUT_NAMES = ['img_1', 'img_2']
    LABEL_NAMES = ['change_map']
    NUM_CLASSES = 2
    CLASS_DIST = {
        'train': [618599552, 167833360],
        'val': [205706240, 56437744],
        'test': [200322672, 61820912],
    }
    SHA1SUM = "5e0fa34b0fec61665b62b622da24f17020ec0664"

    def _init_annotations(self) -> None:
        get_files = lambda name: sorted(glob.glob(os.path.join(self.data_root, self.split, name, "*.png")))
        img_1_filepaths = get_files('time1')
        img_2_filepaths = get_files('time2')
        change_map_filepaths = get_files('label')
        assert len(img_1_filepaths) == len(img_2_filepaths) == len(change_map_filepaths)
        self.annotations = []
        for img_1_path, img_2_path, change_map_path in zip(img_1_filepaths, img_2_filepaths, change_map_filepaths):
            assert all(os.path.basename(x) == os.path.basename(change_map_path) for x in [img_1_path, img_2_path])
            self.annotations.append({
                'img_1_path': img_1_path,
                'img_2_path': img_2_path,
                'change_map_path': change_map_path,
            })

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        inputs = {
            f"img_{input_idx}": utils.io.load_image(
                filepath=self.annotations[idx][f"img_{input_idx}_path"],
                dtype=torch.float32, sub=None, div=255.0,
            ) for input_idx in [1, 2]
        }
        labels = {
            'change_map': utils.io.load_image(
                filepath=self.annotations[idx]['change_map_path'],
                dtype=torch.int64, sub=None, div=255.0,
            )
        }
        meta_info = self.annotations[idx]
        return inputs, labels, meta_info
