from typing import Any, Dict, List, Tuple, Union
import itertools
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from criteria.wrappers import SingleTaskCriterion
from criteria.vision_2d.change_detection.cdmaskformer_criterion.criterion import SetCriterion
from criteria.vision_2d.change_detection.cdmaskformer_criterion.matcher import HungarianMatcher


class CDMaskFormerCriterion(SingleTaskCriterion):
    def __init__(
        self,
        class_weight=2.0,
        dice_weight=5.0,
        mask_weight=5.0,
        no_object_weight=0.1,
        dec_layers = 10,
        num_classes = 1,
    ):
        super(CDMaskFormerCriterion, self).__init__()
        self.class_weight = class_weight
        self.dice_weight = dice_weight
        self.mask_weight = mask_weight
        self.no_object_weight = no_object_weight
        self.dec_layers = dec_layers
        self.num_classes = num_classes

    def __call__(self, y_pred, y_true):
        # building criterion
        matcher = HungarianMatcher(
            cost_class=self.class_weight,
            cost_mask=self.mask_weight,
            cost_dice=self.dice_weight,
            num_points=12544,
        )

        weight_dict = {"loss_ce": self.class_weight, "loss_mask": self.mask_weight, "loss_dice": self.dice_weight}
        aux_weight_dict = {}
        for i in range(self.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]
        criterion = SetCriterion(
            num_classes=self.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=self.no_object_weight,
            losses=losses,
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
        )

        y_pred["pred_masks"]= F.interpolate(
            y_pred["pred_masks"],
            scale_factor=(4, 4),
            mode="bilinear",
            align_corners=False,
        )

        for v in y_pred['aux_outputs']:
            v['pred_masks'] = F.interpolate(
            v["pred_masks"],
            scale_factor=(4, 4),
            mode="bilinear",
            align_corners=False,
        )

        losses = criterion(y_pred, y_true)
        weight_dict = criterion.weight_dict
                    
        loss_ce = 0.0
        loss_dice = 0.0
        loss_mask = 0.0
        for k in list(losses.keys()):
            if k in weight_dict:
                losses[k] *= criterion.weight_dict[k]
                if '_ce' in k:
                    loss_ce += losses[k]
                elif '_dice' in k:
                    loss_dice += losses[k]
                elif '_mask' in k:
                    loss_mask += losses[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)
        loss = loss_ce + loss_dice + loss_mask
        return loss
