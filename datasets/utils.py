from typing import Dict, Callable
import torch


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
