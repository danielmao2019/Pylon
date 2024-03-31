from typing import Dict, Callable
import torch


def apply_transforms(
    transforms: Dict[str, Callable[[torch.Tensor], torch.Tensor]],
    example: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    common_keys = set(transforms.keys()) & set(example.keys())
    for key in common_keys:
        try:
            example[key] = transforms[key](example[key])
        except Exception as e:
            raise RuntimeError(f"[ERROR] Applying transform for {key}: {e}")
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
