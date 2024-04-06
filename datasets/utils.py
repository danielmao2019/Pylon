from typing import List, Dict, Callable, Any
import os
import numpy
import torch
from PIL import Image
from utils.ops import transpose_buffer


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


def apply_transforms(
    transforms: Dict[str, Callable[[torch.Tensor], torch.Tensor]],
    example: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    assert set(example.keys()) == set(['inputs', 'labels', 'meta_info'])
    for key1 in example:
        for key2 in example[key1]:
            if key2 in transforms:
                try:
                    example[key1][key2] = transforms[key2](example[key1][key2])
                except Exception as e:
                    raise RuntimeError(f"[ERROR] Apply transforms['{key2}'] on example['{key1}']['{key2}']: {e}")
    return example


def collate_fn(examples: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    result = transpose_buffer(examples)
    for key1 in result:
        result[key1] = transpose_buffer(result[key1])
        for key2 in result[key1]:
            try:
                result[key1][key2] = torch.stack(result[key1][key2], dim=0)
            except:
                pass
    return result


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
