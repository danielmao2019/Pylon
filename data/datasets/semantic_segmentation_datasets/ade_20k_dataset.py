from typing import Tuple, Dict, Any
import os
import glob
import torch
import json
from data.datasets.base_dataset import BaseDataset
from utils.io.image import load_image


class ADE20KDataset(BaseDataset):
    __doc__ = r"""
    Reference:
        https://github.com/CSAILVision/ADE20K/blob/main/utils/utils_ade20k.py

    Download:
        https://ade20k.csail.mit.edu/

    Used in:
        LoftUp: Learning a Coordinate-Based Feature Upsampler for Vision Foundation Models
    """

    def _init_annotations(self) -> None:
        datapoint_ids = sorted(filter(
            lambda x: os.path.isdir(x),
            glob.glob(os.path.join(
            self.data_root, 'ADE', self.split, "**", "**", "*",
        ))))
        self.annotations = list(map(
            lambda x: {
                'image_filepath': x + ".jpg",
                'object_mask_filepath': x + "_seg.png",
                'parts_masks_filepaths': sorted(glob.glob(x + "_parts_*.png")),
                'attr_filepath': x + ".json",
            }, datapoint_ids
        ))

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        inputs = {
            'image': load_image(
                filepath=self.annotations[idx]['image_filepath'],
                dtype=torch.float32, sub=None, div=255.0,
            ),
        }
        labels = {
            **self._load_object_mask(idx),
            **self._load_parts_masks(idx),
            **self._load_objects_parts(idx),
        }
        meta_info = self.annotations[idx]
        return inputs, labels, meta_info

    def _load_object_mask(self, idx: int) -> torch.Tensor:
        filepath = self.annotations[idx]['object_mask_filepath']
        object_mask = load_image(
            filepath=filepath, dtype=torch.int64,
            sub=None, div=None,
        )

        # Obtain the segmentation mask, built from the RGB channels of the _seg file
        R = object_mask[0, :, :]
        G = object_mask[1, :, :]
        B = object_mask[2, :, :]
        object_cls_mask = (R/10).to(torch.int64) * 256 + G

        # Obtain the instance mask from the blue channel of the _seg file
        m_instances_hat = torch.unique(B, return_inverse=True)[1]
        m_instances_hat = torch.reshape(m_instances_hat, B.shape)
        object_ins_mask = m_instances_hat

        return {
            'object_cls_mask': object_cls_mask,
            'object_ins_mask': object_ins_mask,
        }

    def _load_parts_masks(self, idx: int) -> Dict[str, torch.Tensor]:
        filepaths = self.annotations[idx]['parts_masks_filepaths']
        parts_cls_masks = []
        parts_ins_masks = []
        for filepath in filepaths:
            parts_mask = load_image(
                filepath=filepath, dtype=torch.int64,
                sub=None, div=None,
            )
            R = parts_mask[0, :, :]
            G = parts_mask[1, :, :]
            B = parts_mask[2, :, :]
            parts_cls_masks.append((R/10).to(torch.int64) * 256 + G)
            parts_ins_masks.append(parts_cls_masks[-1])
            # TODO:  correct partinstancemasks
        return {
            'parts_cls_masks': parts_cls_masks,
            'parts_ins_masks': parts_ins_masks,
        }

    def _load_objects_parts(self, idx: int) -> Dict[str, Any]:
        attr_file_name = self.annotations[idx]['attr_filepath']
        with open(attr_file_name, 'r') as f:
            input_info = json.load(f)
        objects = {}
        parts = {}
        contents = input_info['annotation']['object']
        instance = torch.tensor([int(x['id']) for x in contents])
        names = [x['raw_name'] for x in contents]
        corrected_raw_name =  [x['name'] for x in contents]
        partlevel = torch.tensor([int(x['parts']['part_level']) for x in contents])
        ispart = torch.tensor([p > 0 for p in partlevel])
        iscrop = torch.tensor([int(x['crop']) for x in contents])
        listattributes = [x['attributes'] for x in contents]
        polygon = [x['polygon'] for x in contents]
        for p in polygon:
            p['x'] = torch.tensor(p['x'])
            p['y'] = torch.tensor(p['y'])

        objects['instancendx'] = instance[ispart == 0]
        objects['class'] = [names[x] for x in list(torch.where(ispart == 0)[0])]
        objects['corrected_raw_name'] = [corrected_raw_name[x] for x in list(torch.where(ispart == 0)[0])]
        objects['iscrop'] = iscrop[ispart == 0]
        objects['listattributes'] = [listattributes[x] for x in list(torch.where(ispart == 0)[0])]
        objects['polygon'] = [polygon[x] for x in list(torch.where(ispart == 0)[0])]

        parts['instancendx'] = instance[ispart == 1]
        parts['class'] = [names[x] for x in list(torch.where(ispart == 1)[0])]
        parts['corrected_raw_name'] = [corrected_raw_name[x] for x in list(torch.where(ispart == 1)[0])]
        parts['iscrop'] = iscrop[ispart == 1]
        parts['listattributes'] = [listattributes[x] for x in list(torch.where(ispart == 1)[0])]
        parts['polygon'] = [polygon[x] for x in list(torch.where(ispart == 1)[0])]

        return {
            'objects': objects,
            'parts': parts,
        }
