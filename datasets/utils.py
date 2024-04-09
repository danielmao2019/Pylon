import os
import numpy
import torch
from PIL import Image


def load_image(filepath: str, dtype: torch.dtype) -> torch.Tensor:
    assert type(filepath) == str, f"{type(filepath)=}"
    assert os.path.isfile(filepath), f"{filepath=}"
    assert type(dtype) == torch.dtype, f"{type(dtype)=}"
    assert dtype in [torch.float32, torch.uint8], f"{dtype=}"
    image = Image.open(filepath)
    assert image.mode == "RGB"
    image = torch.tensor(numpy.array(image)).permute(2, 0, 1)
    assert len(image.shape) == 3 and image.shape[0] == 3, f"{image.shape=}"
    assert image.dtype == torch.uint8, f"{image.dtype=}, {filepath=}"
    if dtype == torch.float32:
        image = image.type(torch.float32) / 255
    return image


import pycocotools.mask as maskUtils


def poly2mask(mask_ann, img_h, img_w):
    """Private function to convert masks represented with polygon to
    bitmaps.

    Args:
        mask_ann (list | dict): Polygon mask annotation input.
        img_h (int): The height of output mask.
        img_w (int): The width of output mask.

    Returns:
        numpy.ndarray: The decode bitmap mask of shape (img_h, img_w).
    """

    if isinstance(mask_ann, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
    else:
        # rle
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask
