from typing import Dict, Union
import torch
import torchvision
from .base_metric import BaseMetric
from utils.input_checks import check_write_file, check_semantic_segmentation
from utils.io import save_json


class SemanticSegmentationMetric(BaseMetric):

    def __init__(self, num_classes: int, ignore_index: int) -> None:
        super(SemanticSegmentationMetric, self).__init__()
        self.num_classes = num_classes
        self.ignore_index= ignore_index

    def __call__(
        self,
        y_pred: torch.Tensor,
        y_true: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        r"""
        Args:
            y_pred (torch.Tensor): a float32 tensor of shape (N, C, H, W) for predicted logits.
            y_true (torch.Tensor or Dict[str, torch.Tensor]): an int64 tensor of shape (N, H, W) for ground-truth mask.
                If a dictionary is provided, then it's length is assumed to be 1 and the only value is taken as ground truth.

        Return:
            score (torch.Tensor): 1D tensor of length `self.num_classes` representing the IoU scores for each class.
        """
        # input checks
        if type(y_true) == dict:
            assert len(y_true) == 1, f"{y_true.keys()=}"
            y_true = list(y_true.values())[0]
        check_semantic_segmentation(y_pred=y_pred, y_true=y_true)
        # match resolution
        if y_pred.shape[-2:] != y_true.shape[-2:]:
            y_pred = torchvision.transforms.Resize(
                size=y_true.shape[-2:], interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST,
            )(y_pred)
        # make prediction from output
        y_pred = torch.argmax(y_pred, dim=1).type(torch.int64)
        assert y_pred.shape == y_true.shape, f"{y_pred.shape=}, {y_true.shape=}"
        # compute confusion matrix
        valid_mask = y_true != self.ignore_index
        assert valid_mask.sum() >= 1
        y_pred = y_pred[valid_mask]
        y_true = y_true[valid_mask]
        count = torch.bincount(
            y_true * self.num_classes + y_pred, minlength=self.num_classes**2,
        ).view((self.num_classes,) * 2)
        # compute intersection over union
        intersection = count.diag()
        union = count.sum(dim=0, keepdim=False) + count.sum(dim=1, keepdim=False) - count.diag()
        score: torch.Tensor = intersection / union
        # stabilize nan values
        nan_mask = torch.ones(size=(self.num_classes,), dtype=torch.bool, device=count.device)
        nan_mask[y_true.unique()] = False
        assert torch.all(torch.logical_or(score[nan_mask] == 0, torch.isnan(score[nan_mask]))), \
            f"{score.tolist()=}, {nan_mask.tolist()=}, {(score[nan_mask] == 0)=}, {torch.isnan(score[nan_mask])=}"
        score[nan_mask] = float('nan')
        assert score.shape == (self.num_classes,), f"{score.shape=}, {self.num_classes=}"
        # log score
        self.buffer.append(score)
        return score

    def summarize(self, output_path: str = None) -> Dict[str, torch.Tensor]:
        r"""This functions summarizes the semantic segmentation evaluation results on all examples
        seen so far into a single floating point number.
        """
        if output_path is not None:
            check_write_file(path=output_path)
        result: Dict[str, torch.Tensor] = {}
        if len(self.buffer) != 0:
            score = torch.stack(self.buffer, dim=0)
            assert score.shape == (len(self.buffer), self.num_classes), f"{score.shape=}"
            # log IoU per class
            score = torch.nanmean(score, dim=0)
            assert score.shape == (self.num_classes,), f"{score.shape=}"
            result['IoU_per_class'] = score
            # log IoU average
            score = torch.nanmean(score)
            assert score.shape == (), f"{score.shape=}"
            result['IoU_average'] = score
            # log reduction
            assert 'reduced' not in result, f"{result.keys()=}"
            result['reduced'] = result['IoU_average']
        # save to disk
        if output_path is not None:
            save_json(obj=result, filepath=output_path)
        return result
