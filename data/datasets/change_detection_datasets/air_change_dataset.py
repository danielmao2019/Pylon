from typing import Tuple, List, Dict, Any
import itertools
import os
import glob
import random
import torch
import data
from data.datasets import BaseDataset
import utils


class AirChangeDataset(BaseDataset):
    __doc__ = r"""
    Download:
        ```bash
            wget http://mplab.sztaki.hu/~bcsaba/test/SZTAKI_AirChange_Benchmark.zip
            unzip SZTAKI_AirChange_Benchmark.zip
            mv SZTAKI_AirChange_Benchmark AirChange
            rm SZTAKI_AirChange_Benchmark.zip
        ```

    Used in:
        * Change Detection Based on Deep Siamese Convolutional Network for Optical Aerial Images
        * Fully Convolutional Siamese Networks for Change Detection
    """

    SPLIT_OPTIONS = ['train', 'test']
    DATASET_SIZE = {
        'train': 3744,
        'test': 12,
    }
    INPUT_NAMES = ['img_1', 'img_2']
    LABEL_NAMES = ['change_map']
    TRAIN_CROPS_PER_IMAGE = int(3744 / 12)
    IMAGE_SIZE = (952, 640)  # (width, height)
    TEST_CROP_SIZE = (784, 448)  # (width, height)
    TRAIN_CROP_SIZE = (112, 112)  # (width, height)
    NUM_CLASSES = 2
    CLASS_DIST = {
        'train': [44735488, 2229233],
        'test': [4016897, 197887],
    }

    def __init__(self, *args, **kwargs) -> None:
        super(AirChangeDataset, self).__init__(*args, **kwargs)
        self.CLASS_DIST = self.CLASS_DIST[self.split]

    def _init_cropping_configs(self) -> None:

        # Define regions of the L shape (x_start, x_end, y_start, y_end)
        vertical_region = (
            self.TEST_CROP_SIZE[0], self.IMAGE_SIZE[0] - self.TRAIN_CROP_SIZE[0],
            0, self.TEST_CROP_SIZE[1],
        )
        horizontal_region = (
            0, self.TEST_CROP_SIZE[0],
            self.TEST_CROP_SIZE[1], self.IMAGE_SIZE[1] - self.TRAIN_CROP_SIZE[1],
        )
        joint_region = (
            self.TEST_CROP_SIZE[0], self.IMAGE_SIZE[0] - self.TRAIN_CROP_SIZE[0],
            self.TEST_CROP_SIZE[1], self.IMAGE_SIZE[1] - self.TRAIN_CROP_SIZE[1],
        )

        # Precompute valid indices in the L-shaped region
        self.train_crop_locs: List[Tuple[int, int]] = [
            *itertools.product(
                range(vertical_region[0], vertical_region[1]),
                range(vertical_region[2], vertical_region[3])
            ),
            *itertools.product(
                range(horizontal_region[0], horizontal_region[1]),
                range(horizontal_region[2], horizontal_region[3])
            ),
            *itertools.product(
                range(joint_region[0], joint_region[1]),
                range(joint_region[2], joint_region[3])
            ),
        ]

    def _init_annotations(self) -> None:
        # Specify folders
        folders = ["Szada", "Tiszadob"]

        # Collect file paths from both folders
        img_1_filepaths = []
        img_2_filepaths = []
        change_map_filepaths = []

        # Helper function to filter file extensions case-insensitively
        def filter_case_insensitive(filepaths, pattern):
            return [f for f in filepaths if f.lower().endswith(pattern.lower())]

        for folder in folders:
            # Collect all matching files
            img_1_raw = glob.glob(os.path.join(self.data_root, folder, "**", "*im1.*"), recursive=True)
            img_2_raw = glob.glob(os.path.join(self.data_root, folder, "**", "*im2.*"), recursive=True)
            change_map_raw = glob.glob(os.path.join(self.data_root, folder, "**", "*gt.*"), recursive=True)

            # Filter for specific extensions (case-insensitive)
            img_1_filepaths.extend(filter_case_insensitive(img_1_raw, ".bmp"))
            img_2_filepaths.extend(filter_case_insensitive(img_2_raw, ".bmp"))
            change_map_filepaths.extend(filter_case_insensitive(change_map_raw, ".bmp"))

        # Sort the filepaths
        img_1_filepaths.sort()
        img_2_filepaths.sort()
        change_map_filepaths.sort()

        self._init_cropping_configs()
        self.annotations = []
        for img_1_path, img_2_path, change_map_path in zip(img_1_filepaths, img_2_filepaths, change_map_filepaths):
            if self.split == "test":
                # Testing crops: one crop at the top-left corner
                self.annotations.append({
                    "img_1_filepath": img_1_path,
                    "img_2_filepath": img_2_path,
                    "change_map_filepath": change_map_path,
                    "crop_loc": (0, 0),
                    "crop_size": self.TEST_CROP_SIZE,
                })
            elif self.split == "train":
                # Training crops: randomly sample crop locations
                crop_locs = random.choices(self.train_crop_locs, k=self.TRAIN_CROPS_PER_IMAGE)
                for loc in crop_locs:
                    self.annotations.append({
                        "img_1_filepath": img_1_path,
                        "img_2_filepath": img_2_path,
                        "change_map_filepath": change_map_path,
                        "crop_loc": loc,
                        "crop_size": self.TRAIN_CROP_SIZE,
                    })
            else:
                raise ValueError(f"Invalid split: {self.split}. Expected 'train' or 'test'.")

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        inputs = {
            f'img_{input_idx}': utils.io.load_image(
                filepath=self.annotations[idx][f'img_{input_idx}_filepath'],
                dtype=torch.float32, sub=None, div=255.0,
            ) for input_idx in [1, 2]
        }
        labels = {
            'change_map': utils.io.load_image(
                filepath=self.annotations[idx]['change_map_filepath'],
                dtype=torch.int64, sub=None, div=255.0,
            )
        }
        meta_info = {
            'img_1_filepath': self.annotations[idx]['img_1_filepath'],
            'img_2_filepath': self.annotations[idx]['img_2_filepath'],
            "change_map_filepath": self.annotations[idx]['change_map_filepath'],
            'image_size': self.IMAGE_SIZE,
            'crop_loc': self.annotations[idx]['crop_loc'],
            'crop_size': self.annotations[idx]['crop_size'],
        }
        crop_op = data.transforms.Crop(loc=meta_info['crop_loc'], size=meta_info['crop_size'])
        inputs = {key: crop_op(inputs[key]) for key in inputs}
        labels = {key: crop_op(labels[key]) for key in labels}
        return inputs, labels, meta_info
