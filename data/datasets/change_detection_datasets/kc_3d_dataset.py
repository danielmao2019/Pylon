from typing import Tuple, Dict, Any, Optional
import os
import pickle
import numpy
import torch
from torchvision.ops import masks_to_boxes
from einops import rearrange
import utils


class KC3DDataset:
    __doc__ = r"""
    Reference:
        * https://github.com/ragavsachdeva/CYWS-3D/blob/master/kc3d.py

    Download:
        * do the following
        ```
        # download
        mkdir <data-root>
        cd <data-root>
        wget https://thor.robots.ox.ac.uk/cyws-3d/kc3d.tar
        # extract
        tar xvf kc3d.tar
        ```

    Used in:
        * The Change You Want to See (Now in 3D)
    """

    SPLIT_OPTIONS = ['train', 'val', 'test']
    DATASET_SIZE = None  # Define if exact sizes for splits are known
    INPUT_NAMES = ['img_1', 'img_2']
    LABEL_NAMES = ['bbox_1', 'bbox_2']
    SHA1SUM = None  # Define if checksum validation is required

    def __init__(self, use_ground_truth_registration: Optional[bool] = True) -> None:
        super(KC3DDataset, self).__init__()
        assert isinstance(use_ground_truth_registration, bool)
        self.use_ground_truth_registration = use_ground_truth_registration

    def _init_annotations_(self, split: Optional[str]) -> None:
        # Path to dataset split file
        split_file_path = os.path.join(self.data_root, "data_split.pkl")
        if not os.path.exists(split_file_path):
            raise FileNotFoundError(f"Split file not found at {split_file_path}.")

        # Load split data
        with open(split_file_path, "rb") as file:
            dataset_splits = pickle.load(file)
        assert split in dataset_splits
        
        # Load annotations for the specified split
        self.annotations = dataset_splits[split]

    def get_target_bboxes_from_mask(self, mask_as_tensor):
        """
        Converts a mask tensor to bounding boxes.
        """
        if len(mask_as_tensor.shape) == 2:
            mask_as_tensor = rearrange(mask_as_tensor, "h w -> 1 h w")
        bboxes = masks_to_boxes(mask_as_tensor)
        return bboxes

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        """
        Combines inputs, labels, and meta_info for a single datapoint.
        """
        inputs = self._load_inputs(idx)
        labels = self._load_labels(idx)
        meta_info = self._load_meta_info()
        return inputs, labels, meta_info

    def _load_inputs(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load input data: images, depth maps, and metadata.
        """
        # Load RGB images
        img_1 = utils.io.load_image(
            filepath=self.annotation[idx]['image1'],
            dtype=torch.float32, sub=None, div=255.0,
        )
        img_2 = utils.io.load_image(
            filepath=self.annotation[idx]['image2'],
            dtype=torch.float32, sub=None, div=255.0,
        )

        # Load depth maps
        depth_1 = utils.io.load_image(
            filepaths=[self.annotation[idx]['depth1']],
            dtype=torch.float32, sub=None, div=None,
        )
        depth_2 = utils.io.load_image(
            filepaths=[self.annotation[idx]['depth2']],
            dtype=torch.float32, sub=None, div=None,
        )

        # Initialize inputs
        inputs = {
            'img_1': img_1,
            'img_2': img_2,
            'depth_1': depth_1,
            'depth_2': depth_2,
        }

        # Add ground truth registration metadata if enabled
        if self.use_ground_truth_registration:
            metadata_file = os.path.join(
                self.data_root,
                "_".join(self.annotations[idx]["image1"].split(".")[0].split("_")[:3]) + ".npy",
            )
            metadata = numpy.load(metadata_file, allow_pickle=True).item()

            for key, value in metadata.items():
                if key == "intrinsics":
                    inputs["intrinsics1"] = torch.Tensor(value)
                    inputs["intrinsics2"] = torch.Tensor(value)
                elif key == "position_before":
                    inputs["position1"] = torch.Tensor(value)
                elif key == "position_after":
                    inputs["position2"] = torch.Tensor(value)
                elif key == "rotation_before":
                    inputs["rotation1"] = torch.Tensor(value)
                elif key == "rotation_after":
                    inputs["rotation2"] = torch.Tensor(value)

        return inputs

    def _load_labels(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load labels: bounding boxes extracted from masks.
        """
        # Load masks
        mask_1 = utils.io.load_image(
            filepath=os.path.join(self.data_root, self.annotation[idx]["mask1"]),
            dtype=torch.int64, sub=None, div=None,
        )
        mask_2 = utils.io.load_image(
            filepath=os.path.join(self.data_root, self.annotation[idx]["mask2"]),
            dtype=torch.int64, sub=None, div=None,
        )

        # Convert masks to bounding boxes
        bbox_1 = self.get_target_bboxes_from_mask(mask_1)
        bbox_2 = self.get_target_bboxes_from_mask(mask_2)

        labels = {
            'bbox_1': bbox_1.tolist(),
            'bbox_2': bbox_2.tolist(),
        }
        return labels

    def _load_meta_info(self):
        """
        Load meta information for the datapoint.
        """
        return {'registration_strategy': '3d'}
