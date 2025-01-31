from typing import Tuple, Dict, Any
import os
import glob
import random
import torch
import matplotlib.pyplot as plt
from data.datasets import BaseDataset
import utils
from utils.input_checks.str_types import check_write_dir


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

    def visualize(self, output_dir: str) -> None:
        check_write_dir(output_dir)
        random_indices = random.sample(population=range(len(self)), k=10)

        for idx in random_indices:
            datapoint = self.__getitem__(idx)
            inputs, labels = datapoint['inputs'], datapoint['labels']

            img_1 = inputs['img_1']  # (C, H, W)
            img_2 = inputs['img_2']  # (C, H, W)
            change_map = labels['change_map']  # (H, W)

            # Convert tensors to numpy format
            img_1 = (img_1.permute(1, 2, 0) * 255).type(torch.uint8).cpu().numpy()  # (H, W, C)
            img_2 = (img_2.permute(1, 2, 0) * 255).type(torch.uint8).cpu().numpy()  # (H, W, C)
            change_map = (change_map * 255).cpu().numpy()  # (H, W)

            # Create a figure
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_1)
            axes[0].set_title("Image 1")
            axes[0].axis("off")

            axes[1].imshow(img_2)
            axes[1].set_title("Image 2")
            axes[1].axis("off")

            axes[2].imshow(change_map, cmap="gray")
            axes[2].set_title("Change Map")
            axes[2].axis("off")

            # Save the figure
            save_path = os.path.join(output_dir, f"datapoint_{idx}.png")
            plt.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
