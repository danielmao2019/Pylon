from typing import Tuple, List, Dict, Any
import os
import torch
from data.datasets import BaseDataset
import utils

class CDDDataset(BaseDataset):
    """
    The CDDDataset class is designed for managing and loading data for change detection tasks.

    References:
        * 

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

    Attributes:
        SPLIT_OPTIONS (List[str]): Available data splits ('train', 'test', 'val').
        DATASET_SIZE (Optional[int]): Dataset size (set dynamically).
        INPUT_NAMES (List[str]): Names of input tensors.
        LABEL_NAMES (List[str]): Names of label tensors.
        SHA1SUM (Optional[str]): Checksum for data validation.
    """

    SPLIT_OPTIONS = ['train', 'test', 'val']
    DATASET_SIZE = None
    INPUT_NAMES = ['img_1', 'img_2']
    LABEL_NAMES = ['change_map']
    SHA1SUM = None

    def _init_annotations_(self, split: str) -> None:
        """
        Initialize dataset annotations by parsing directory structure and
        associating file paths with inputs and labels.

        Args:
            split (str): The data split ('train', 'test', 'val').

        Raises:
            AssertionError: If expected files or directories are missing.
        """
<<<<<<< HEAD
        model_path = os.path.join(self.data_root, 'Model')
        subfolders = os.listdir(model_path)
        inputs_root = [os.path.join(model_path, subfolder, split) for subfolder in subfolders]
        labels_root = [os.path.join(model_path, subfolder, split, 'OUT') for subfolder in subfolders]

        self.annotations: List[dict] = []
        for input_root, label_root in zip(inputs_root, labels_root):
            filenames = [file for file in os.listdir(os.path.join(input_root, 'A')) if not file.startswith('.')]
            filenames.sort()

            for filename in filenames:
                input_1_filepath = os.path.join(input_root, 'A', filename)
                input_2_filepath = os.path.join(input_root, 'B', filename)
                label_filepath = os.path.join(label_root, filename)

                # Ensure required files exist
                assert os.path.isfile(input_1_filepath), f"File not found: {input_1_filepath}"
                assert os.path.isfile(input_2_filepath), f"File not found: {input_2_filepath}"
                assert os.path.isfile(label_filepath), f"File not found: {label_filepath}"

=======
        subfolders = (os.listdir(os.path.join(self.data_root, 'Model')))
        inputs_root: str = [os.path.join(self.data_root, 'Model', f"{subfolder}", f"{split}") for subfolder in subfolders]
        labels_root: str = [os.path.join(self.data_root, 'Model', f"{subfolder}",  f"{split}", 'OUT') for subfolder in subfolders]
        self.annotations: List[dict] = []
        for input_root, label_root in zip(inputs_root, labels_root):
            filenames = [files for files in os.listdir(os.path.join(input_root, 'A')) if not files.startswith('.')]
            filenames.sort()
            print(filenames)
            assert len(os.listdir(os.path.join(input_root, 'A'))), len(os.listdir(os.path.join(input_root, 'B')))
            for filename in filenames:
                input_1_filepath = os.path.join(input_root, 'A', filename)
                assert os.path.isfile(input_1_filepath), f"{input_1_filepath=}"
                input_2_filepath = os.path.join(input_root, 'B', filename)
                assert os.path.isfile(input_1_filepath), f"{input_1_filepath=}"
                label_filepath = os.path.join(label_root, filename)
                assert os.path.isfile(label_filepath), f"{label_filepath=}"
>>>>>>> f
                self.annotations.append({
                    'inputs': {
                        'input_1_filepath': input_1_filepath,
                        'input_2_filepath': input_2_filepath,
                    },
                    'labels': {
                        'label_filepath': label_filepath,
                    },
                })

    def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Load a single datapoint by index.

        Args:
            idx (int): Index of the datapoint to load.

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
                - A dictionary containing input tensors.
                - A dictionary containing label tensors.
                - A dictionary containing metadata.
        """
        inputs = self._load_inputs(idx)
        labels = self._load_labels(idx)
        height, width = labels['change_map'].shape

        assert all(
            x.shape == (3, height, width) for x in [inputs['img_1'], inputs['img_2']]
        ), f"Shape mismatch: {inputs['img_1'].shape}, {inputs['img_2'].shape}, {labels['change_map'].shape}"

        meta_info = {'image_resolution': (height, width)}
        return inputs, labels, meta_info

    def _load_inputs(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load input images for a given index.

        Args:
            idx (int): Index of the datapoint.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the input tensors.
        """
        inputs = {}
        for input_idx in [1, 2]:
            img = utils.io.load_image(
                filepath=self.annotations[idx]['inputs'][f'input_{input_idx}_filepath'],
                dtype=torch.float32, sub=None, div=255.0,
            )
            inputs[f'img_{input_idx}'] = img
        return inputs

    def _load_labels(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load the label (change map) for a given index.

        Args:
            idx (int): Index of the datapoint.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the label tensor.

        Raises:
            AssertionError: If the loaded label is not 2D.
        """
        change_map = utils.io.load_image(
            filepath=self.annotations[idx]['labels']['label_filepath'],
            dtype=torch.int64, sub=None, div=255.0,
        )
        assert change_map.ndim == 2, f"Expected 2D label, got {change_map.shape}."
        return {'change_map': change_map}
