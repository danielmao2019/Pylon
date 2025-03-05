import glob
import os
import torch
from data.datasets import BaseDataset
import utils
from PIL import Image

class WHU_CD_Dataset(BaseDataset):
    __doc__ = r"""
    Download:
        ```bash
            wget http://mplab.sztaki.hu/~bcsaba/test/SZTAKI_AirChange_Benchmark.zip
            unzip SZTAKI_AirChange_Benchmark.zip
            mv SZTAKI_AirChange_Benchmark AirChange
            rm SZTAKI_AirChange_Benchmark.zip
        ```
        
        mv 1.\ The\ two-period\ image\ data/ images
        mv 2.\ The\ shape\ file\ of\ the\ images/ image_shape

    Used in:
        * Change Detection Based on Deep Siamese Convolutional Network for Optical Aerial Images
        * Fully Convolutional Siamese Networks for Change Detection
    """
    SPLIT_OPTIONS = ['train', 'test']
    DATASET_SIZE = {
        'train': 4838,
        'test': 2596,
    }
    INPUT_NAMES = ['img_1', 'img_2']
    LABEL_NAMES = ['change_map']
    IMAGE_SIZE = (255, 255)  # (width, height)
    NUM_CLASSES = 2
    # this is a rough estimate using 5 initializations due to randomness in the dataset
    CLASS_DIST = {
        'train': [302740256, 14323049],
        'test': [163865856, 6265578],
    }
    Image.MAX_IMAGE_PIXELS = None
    
    def _init_annotations(self) -> None:
        # Get TIF file path
        inputs_image1_root: str = os.path.join(self.data_root, 'images', '2012', 'whole_image', self.split)
        inputs_image2_root: str = os.path.join(self.data_root, 'images', '2016', 'whole_image', self.split)
        labels_root: str = os.path.join(self.data_root, 'images', 'change_label', self.split)
        # Split condition
        if 'cropped_images' not in os.listdir(inputs_image1_root):
            self.crop_images(glob.glob(os.path.join(inputs_image1_root, 'label', '*.tif')), os.path.join(inputs_image1_root, 'cropped_images'))
        if 'cropped_images' not in os.listdir(inputs_image2_root):
            self.crop_images(glob.glob(os.path.join(inputs_image2_root, 'label', '*.tif')), os.path.join(inputs_image2_root, 'cropped_images'))
        if 'cropped_images' not in os.listdir(labels_root):
            self.crop_images(glob.glob(os.path.join(labels_root, '*.tif')), os.path.join(labels_root, 'cropped_images'))    
        assert 'cropped_images' in os.listdir(inputs_image1_root)
        assert 'cropped_images' in os.listdir(inputs_image2_root)
        assert 'cropped_images' in os.listdir(labels_root)
        img_1_filepaths = glob.glob(os.path.join(inputs_image1_root, 'cropped_images', '*.png'))
        img_2_filepaths = glob.glob(os.path.join(inputs_image2_root, 'cropped_images', '*.png'))
        change_map_filepaths = glob.glob(os.path.join(labels_root, 'cropped_images', '*.png'))
        assert len(img_1_filepaths) == len(img_2_filepaths)
        assert len(img_1_filepaths) == len(change_map_filepaths)
        self.annotations = []
        for img_1_path, img_2_path, change_map_path in zip(img_1_filepaths, img_2_filepaths, change_map_filepaths):
            assert all(os.path.basename(x) == os.path.basename(change_map_path) for x in [img_1_path, img_2_path])
            self.annotations.append({
                'img_1_path': img_1_path,
                'img_2_path': img_2_path,
                'change_map_path': change_map_path,
            })
        
    def _load_datapoint(self, idx):
        # Load Inputs
        # assertion
        inputs = {
            f"img_{input_idx}": utils.io.load_image(
                filepaths=self.annotations[idx][f"img_{input_idx}_path"],
                dtype=torch.float32,
            ) for input_idx in [1, 2]
        }
        labels = {
            'change_map': utils.io.load_image(
                filepaths=self.annotations[idx]['change_map_path'],
                dtype=torch.int64,
            )
        }
        meta_info = self.annotations[idx]
        return inputs, labels, meta_info

    def crop_images(self, input_dir, output_dir):
        assert len(input_dir) == 1, f'{input_dir}'
        assert os.path.isfile(input_dir[0])
        if not os.path.exists(output_dir):
            image = Image.open(input_dir[0])
            if image.mode == '1':  # 1-bit images are not ideal for further processing
                image = image.convert('L')
            width, height = image.size
            patch_size = 256
            patches = []
            # Loop through the image with a step equal to patch_size (non-overlapping)
            for y in range(0, height - patch_size + 1, patch_size):
                for x in range(0, width - patch_size + 1, patch_size):
                    patch = image.crop((x, y, x + patch_size, y + patch_size))
                    patches.append(patch)
            # Create an output directory for the patches
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # Save each patch as a separate TIFF file named from 0.tif to n.tif
            for idx, patch in enumerate(patches):
                patch_filename = os.path.join(output_dir, f"{idx}.tif")
                patch.save(patch_filename)